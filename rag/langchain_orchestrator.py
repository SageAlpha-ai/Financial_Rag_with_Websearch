"""
LangChain Orchestration Layer - OpenAI-Style Answerability Validation

Follows OpenAI's approach:
1. ALWAYS retrieves documents first
2. Validates answerability (entity, year, metric matching)
3. Only generates RAG answer if documents are answerable
4. Returns RAG_NO_ANSWER if data doesn't match requirements
5. Falls back to LLM only when retrieval fails completely

Uses LangChain v1 LCEL (LangChain Expression Language) pattern.
"""

import contextvars
import logging
import math
import os
import re
import statistics
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urlparse

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from rank_bm25 import BM25Okapi

from config.settings import get_config, DEBUG_MODE
from vectorstore.chroma_client import get_company_collection, list_all_collections

# Web search imports
from rag.web_search import WebSearchEngine
from rag.investor_scraper import InvestorRelationsScraper
from rag.document_extractor import DocumentExtractor
from rag.temp_storage import get_temp_storage
from rag.evidence_fusion import EvidenceFusion
from rag.metrics import MetricsRecorder
from rag.confidence_scorer import ConfidenceScorer
from rag.numeric_validator import NumericValidator
from rag.company_extractor import extract_company_name
from rag.planner import plan_query
from rag.telemetry import cost_tracker_callback, set_llm_cost_stage

logger = logging.getLogger(__name__)

# Optional LangSmith tracing (disabled by default)
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "rag-service")

if LANGCHAIN_TRACING_V2 and LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
    logger.info("LangSmith tracing enabled")

# ---------------------------------------------------------------------------
# Per-request debug flag (set from X-Debug header by the API layer)
# ---------------------------------------------------------------------------
_debug_request: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_debug_request", default=False
)
_request_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "request_id", default=None
)


def set_debug_request(enabled: bool) -> None:
    """Activate per-request debug mode (called by the API layer)."""
    _debug_request.set(enabled)


def set_request_id(rid: Optional[str]) -> None:
    """Set request correlation ID for structured logging (called by the API layer)."""
    _request_id.set(rid)


def get_request_id() -> Optional[str]:
    """Return current request ID if set (for planner/adequacy logs)."""
    return _request_id.get(None)


def _is_debug_enabled() -> bool:
    """Return ``True`` when audit metadata should be attached."""
    return DEBUG_MODE or _debug_request.get(False)


def _load_all_documents_from_chroma(collection) -> Tuple[List[str], List[Dict]]:
    """Load all documents from Chroma collection for BM25 indexing."""
    try:
        all_data = collection.get(include=["documents", "metadatas"])
        documents = all_data.get("documents", [])
        metadatas = all_data.get("metadatas", [])
        logger.info(f"Loaded {len(documents)} documents from Chroma for BM25 indexing")
        return documents, metadatas
    except Exception as e:
        logger.error(f"Failed to load documents from Chroma: {e}")
        return [], []


def _extract_fiscal_year(query: str) -> Optional[str]:
    """Extracts fiscal year from query. Returns normalized FYxxxx format."""
    patterns = [
        r'FY\s*(\d{4})',
        r'fiscal\s+year\s+(\d{4})',
        r'\b(20\d{2})\b',
        r'\b(19\d{2})\b',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            year = match.group(1)
            return f"FY{year}"
    
    return None


# DEPRECATED — replaced by planner-based routing (rag/planner.py).
# This function only recognised three OFSS name variants, making it
# impossible to resolve any other company.  The planner removes the
# need for hard-coded entity extraction at the routing layer.
# Retained for: legacy answer_query() fallback when ENABLE_QUERY_PLANNER=false.
def _extract_entity_from_query(query: str) -> Optional[str]:
    """Extracts company/entity from query.

    .. deprecated:: Replaced by planner-based routing.
    """
    query_lower = query.lower()
    
    entity_mappings = {
        "oracle financial services": "Oracle Financial Services Software Ltd",
        "oracle financial": "Oracle Financial Services Software Ltd",
        "ofss": "Oracle Financial Services Software Ltd",
    }
    
    for key, value in entity_mappings.items():
        if key in query_lower:
            return value
    
    return None


def _extract_metrics_from_query(query: str) -> List[str]:
    """Extracts requested financial metrics from query."""
    metrics = []
    query_lower = query.lower()
    
    metric_mapping = {
        'revenue': ['revenue', 'sales', 'turnover'],
        'net_income': ['net income', 'net profit', 'profit', 'earnings', 'pat'],
        'ebitda': ['ebitda'],
        'gross_profit': ['gross profit'],
        'operating_income': ['operating income', 'operating profit', 'ebit'],
        'assets': ['assets', 'total assets'],
        'equity': ['equity', 'total equity'],
    }
    
    for metric_key, keywords in metric_mapping.items():
        if any(kw in query_lower for kw in keywords):
            metrics.append(metric_key)
    
    return metrics


def _is_answerable(
    question: str,
    rag_docs: List[str],
    rag_metas: List[Dict],
    web_docs: List[Dict],
) -> Tuple[bool, str]:
    """Strict answerability gate — blocks answers built from irrelevant evidence.

    Checks whether retrieved evidence satisfies the constraints implied by the
    query (entity, fiscal year, financial metrics).  Returns ``(False, reason)``
    when a required constraint is violated so the caller can short-circuit with
    a safe NO_ANSWER response.

    Returns ``(True, "")`` when all detected constraints are satisfied **or**
    when no documents were retrieved at all (the LLM-only disclaimer path
    handles that case).
    """
    # If no evidence was retrieved at all, block numeric / financial
    # queries (which require verified data) but allow general-knowledge
    # questions through to the LLM-only disclaimer path.
    if not rag_docs and not web_docs:
        _FINANCIAL_KEYWORDS = [
            "quarterly", "quarter", "q1", "q2", "q3", "q4",
            "annual report", "balance sheet", "income statement",
            "cash flow", "exact figure", "financial statement",
        ]
        has_financial_kw = any(
            kw in question.lower() for kw in _FINANCIAL_KEYWORDS
        )
        if (
            _detect_numeric_intent(question)
            or _extract_fiscal_year(question)
            or _extract_metrics_from_query(question)
            or has_financial_kw
        ):
            logger.warning(
                "[GUARDRAIL] No evidence + numeric/financial intent — "
                "blocking LLM-only answer",
            )
            return (
                False,
                "No verified documents available for numeric or financial query",
            )
        return True, "No evidence required for general knowledge query"

    # --- Extract query constraints ---
    requested_entity = extract_company_name(question)
    requested_year = _extract_fiscal_year(question)
    requested_metrics = _extract_metrics_from_query(question)

    # No specific constraints detected → pass
    if not requested_entity and not requested_year and not requested_metrics:
        return True, ""

    # --- Combine evidence metadata and texts ---
    all_metas: List[Dict] = list(rag_metas)
    all_texts: List[str] = list(rag_docs)
    for wdoc in web_docs:
        all_metas.append(wdoc.get("metadata", {}))
        all_texts.append(wdoc.get("text", ""))

    # --- Entity constraint ---
    if requested_entity:
        entity_found = False
        has_company_metadata = False
        for meta in all_metas:
            company = meta.get("company", "")
            if company:
                has_company_metadata = True
                if (
                    requested_entity.lower() in company.lower()
                    or company.lower() in requested_entity.lower()
                ):
                    entity_found = True
                    break
        # Block only when documents carry company metadata but none match.
        # Web-search results often lack a ``company`` field; in that case
        # the entity constraint is not enforceable and we allow the answer.
        if has_company_metadata and not entity_found:
            logger.warning(
                "[GUARDRAIL] Entity constraint failed: requested='%s', "
                "available companies=%s",
                requested_entity,
                [m.get("company", "") for m in all_metas if m.get("company")],
            )
            return (
                False,
                f"Entity mismatch: no documents match requested company "
                f"'{requested_entity}'",
            )

    # --- Fiscal year constraint ---
    if requested_year:
        year_found = False
        has_year_metadata = False
        for meta in all_metas:
            doc_year = meta.get("fiscal_year", "")
            if doc_year:
                has_year_metadata = True
                if requested_year.lower() == doc_year.lower():
                    year_found = True
                    break
        if has_year_metadata and not year_found:
            logger.warning(
                "[GUARDRAIL] Fiscal year constraint failed: requested='%s', "
                "available years=%s",
                requested_year,
                [m.get("fiscal_year", "") for m in all_metas if m.get("fiscal_year")],
            )
            return (
                False,
                f"Fiscal year mismatch: no documents match requested period "
                f"'{requested_year}'",
            )

    # --- Metric constraint ---
    if requested_metrics:
        metric_found = False
        for text in all_texts:
            text_lower = text.lower()
            for metric in requested_metrics:
                if metric == "revenue" and "revenue" in text_lower:
                    metric_found = True
                elif metric == "net_income" and (
                    "net income" in text_lower or "net profit" in text_lower
                ):
                    metric_found = True
                elif metric in text_lower:
                    metric_found = True
                if metric_found:
                    break
            if metric_found:
                break
        if not metric_found:
            logger.warning(
                "[GUARDRAIL] Metric constraint failed: requested=%s",
                requested_metrics,
            )
            return (
                False,
                f"Metric mismatch: no documents contain requested metrics "
                f"{requested_metrics}",
            )

    return True, ""


# DEPRECATED — replaced by planner-based routing (rag/planner.py).
# This heuristic pre-checked entity/year/metric overlap between the query
# and retrieved documents to decide if RAG could answer.  It relied on the
# same hard-coded keyword and entity maps as the rest of the legacy routing
# chain, causing false negatives when metadata was incomplete or when
# queries used synonyms not in the keyword list.
# The planner delegates answerability to the final LLM step, which reasons
# over the full context rather than string-matching metadata fields.
# Retained for: legacy answer_query() fallback when ENABLE_QUERY_PLANNER=false.
def _validate_answerability(
    query: str,
    documents: List[str],
    metadatas: List[Dict]
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    OpenAI-style answerability validation.

    .. deprecated::
        Replaced by planner-based routing.  When the planner is enabled,
        this function is never called.

    Validates if retrieved documents can answer the query by checking:
    1. Entity match (if entity specified in query)
    2. Fiscal year match (if year specified in query)
    3. Metric match (if metric specified in query)
    
    Returns:
        (is_answerable, reason, validation_details)
    """
    if not documents:
        return False, "No documents retrieved", {}
    
    requested_year = _extract_fiscal_year(query)
    requested_entity = extract_company_name(query)
    requested_metrics = _extract_metrics_from_query(query)
    
    validation_details = {
        "requested_year": requested_year,
        "requested_entity": requested_entity,
        "requested_metrics": requested_metrics,
        "entity_matches": 0,
        "year_matches": 0,
        "metric_matches": 0,
        "strong_matches": 0
    }
    
    # Check each document for matches
    for doc, meta in zip(documents, metadatas):
        doc_lower = doc.lower()
        doc_year = meta.get("fiscal_year", "")
        doc_entity = meta.get("company", "")
        
        # Entity match
        entity_match = False
        if requested_entity:
            if doc_entity and requested_entity.lower() in doc_entity.lower():
                validation_details["entity_matches"] += 1
                entity_match = True
        else:
            # No entity specified, consider it a match
            entity_match = True
        
        # Year match (CRITICAL for financial queries)
        year_match = False
        if requested_year:
            if doc_year and requested_year.lower() == doc_year.lower():
                validation_details["year_matches"] += 1
                year_match = True
        else:
            # No year specified, consider it a match
            year_match = True
        
        # Metric match
        metric_match = False
        if requested_metrics:
            for metric in requested_metrics:
                if metric == "revenue" and "revenue" in doc_lower:
                    validation_details["metric_matches"] += 1
                    metric_match = True
                    break
                elif metric == "net_income" and ("net income" in doc_lower or "net profit" in doc_lower):
                    validation_details["metric_matches"] += 1
                    metric_match = True
                    break
                elif metric in doc_lower:
                    validation_details["metric_matches"] += 1
                    metric_match = True
                    break
        else:
            # No specific metric, consider it a match
            metric_match = True
        
        # Strong match = all requirements met
        if entity_match and year_match and metric_match:
            validation_details["strong_matches"] += 1
    
    # Determine answerability
    is_answerable = False
    reason = ""
    
    if requested_year and validation_details["year_matches"] == 0:
        is_answerable = False
        reason = f"Query requires FY{requested_year[2:]} data, but retrieved documents contain different fiscal years"
    elif requested_entity and validation_details["entity_matches"] == 0:
        is_answerable = False
        reason = f"Query requires {requested_entity} data, but retrieved documents are for different entities"
    elif requested_metrics and validation_details["metric_matches"] == 0:
        is_answerable = False
        reason = f"Query requires {', '.join(requested_metrics)} data, but retrieved documents don't contain this metric"
    elif validation_details["strong_matches"] > 0:
        is_answerable = True
        reason = f"Found {validation_details['strong_matches']} document(s) matching all requirements"
    elif not requested_year and not requested_entity and not requested_metrics:
        # General query, no specific requirements
        is_answerable = True
        reason = "General query with no specific requirements - documents are relevant"
    else:
        is_answerable = False
        reason = "Retrieved documents don't match query requirements"
    
    logger.info(f"[VALIDATE] Answerability check: {is_answerable}")
    logger.info(f"[VALIDATE] Reason: {reason}")
    logger.info(f"[VALIDATE] Details: entity_matches={validation_details['entity_matches']}, "
                f"year_matches={validation_details['year_matches']}, "
                f"metric_matches={validation_details['metric_matches']}, "
                f"strong_matches={validation_details['strong_matches']}")
    
    return is_answerable, reason, validation_details


def _detect_numeric_intent(query: str) -> bool:
    """Detect if query has numeric/exact intent."""
    query_lower = query.lower()
    numeric_patterns = [
        r'\d+', r'%', r'\$', r'\btotal\b', r'\bexact\b', r'\brate\b',
        r'\bvalue\b', r'\brevenue\b', r'\bamount\b', r'\bnumber\b', r'\bcount\b'
    ]
    numeric_score = sum(1 for pattern in numeric_patterns if re.search(pattern, query_lower, re.IGNORECASE))
    return numeric_score >= 2


def _detect_temporal_intent(query: str) -> bool:
    """Detect if query requires real-time or current information.

    Returns ``True`` when the query contains temporal keywords that signal
    the user wants *present-moment* data (e.g. today's date, current price,
    latest revenue).  Simple keyword matching — no LLM call.
    """
    _TEMPORAL_KEYWORDS = [
        "today", "current", "now", "latest", "as of",
        "present", "current time", "current date",
    ]
    query_lower = query.lower()
    return any(kw in query_lower for kw in _TEMPORAL_KEYWORDS)


def _detect_system_introspection(query: str) -> bool:
    """Detect if the query asks about the system's own capabilities.

    Returns ``True`` for questions about web/internet access, browsing
    capability, real-time data access, or training-data scope.  These
    queries are answered with a deterministic system-authored response
    so that no planner, RAG, web search, or LLM call is needed.
    """
    _INTROSPECTION_PHRASES = [
        "do you have internet",
        "do you have web",
        "can you browse",
        "can you search the web",
        "can you search the internet",
        "can you access the internet",
        "can you access the web",
        "can you access websites",
        "do you have access to the internet",
        "do you have access to the web",
        "do you have web search",
        "do you have real-time",
        "do you have realtime",
        "do you have real time",
        "do you have live data",
        "can you access real-time",
        "can you access realtime",
        "can you access live data",
        "what data do you have access to",
        "what sources do you use",
        "what is your training data",
        "what are your capabilities",
        "what can you do",
        "are you connected to the internet",
        "are you connected to the web",
        "do you browse the web",
        "can you look up websites",
        "can you fetch web pages",
        "can you go online",
        "are you online",
        "training data",
        "training cutoff",
        "knowledge cutoff",
    ]
    query_lower = query.lower()
    return any(phrase in query_lower for phrase in _INTROSPECTION_PHRASES)


_SYSTEM_CAPABILITY_RESPONSE = (
    "SageAlpha is a financial analysis system with the following capabilities:\n\n"
    "1. **Internal Document Retrieval (RAG)** — queries are matched against a "
    "curated knowledge base of ingested financial documents, annual reports, "
    "and regulatory filings using semantic and keyword search.\n\n"
    "2. **Curated Web Search** — when internal documents are insufficient, "
    "the system performs a targeted web search restricted to trusted, "
    "authoritative domains (e.g. RBI, NSE, BSE, and other government sources). "
    "Results from untrusted or general-purpose sites are automatically filtered "
    "out.\n\n"
    "3. **LLM-Powered Analysis** — retrieved evidence is synthesised by a "
    "large language model to produce clear, cited answers.\n\n"
    "These components work together to deliver accurate, source-backed "
    "financial insights."
)


def _resolve_rag_title(meta: Dict) -> str:
    """Resolve a single RAG metadata item to an audit-grade title.

    Priority order:
        1) title
        2) document_title
        3) file_name (or filename)
        4) source
        5) collection_name
        6) company + " Financial Report"
        7) "Internal Financial Document"

    Empty strings and None are treated as missing. If metadata contains
    page_number or page, appends " (Page N)".
    """
    def _non_empty(val: Any) -> bool:
        return val is not None and str(val).strip() != ""

    candidates = [
        meta.get("title"),
        meta.get("document_title"),
        meta.get("file_name") or meta.get("filename"),
        meta.get("source"),
        meta.get("collection_name"),
    ]
    for c in candidates:
        if _non_empty(c):
            title = str(c).strip()
            break
    else:
        company = meta.get("company")
        if _non_empty(company):
            title = f"{str(company).strip()} Financial Report"
        else:
            title = "Internal Financial Document"

    page = meta.get("page_number") or meta.get("page")
    if page is not None and str(page).strip() != "":
        title = f"{title} (Page {page})"

    return title


def _normalize_sources(
    rag_metas: List[Dict],
    web_docs: List[Dict],
    execution_path: str,
) -> List[Dict]:
    """Return a consistently-shaped ``sources`` list for any execution path.

    Every item in the returned list conforms to::

        {
            "type":    "rag" | "web" | "system",
            "title":   str,
            "url":     str | None,
        }

    RAG sources never have None or empty title; title is resolved via
    _resolve_rag_title() for audit-grade traceability. Snippet is omitted
    for RAG (internal documents). URL remains None for internal RAG docs.
    """
    sources: List[Dict] = []

    include_rag = "RAG" in execution_path
    include_web = "WEB_SEARCH" in execution_path

    if execution_path == "SYSTEM":
        sources.append({
            "type": "system",
            "title": "System Capability Declaration",
            "url": None,
            "snippet": "This response describes system capabilities.",
        })
        logger.debug("[PLANNER] sources normalized for execution_path=%s", execution_path)
        return sources

    if execution_path == "LLM_ONLY":
        logger.debug("[PLANNER] sources normalized for execution_path=%s", execution_path)
        return []

    # --- RAG sources (audit-grade title, no snippet) ---
    if include_rag and rag_metas:
        for meta in rag_metas:
            title = _resolve_rag_title(meta)
            logger.debug("[SOURCE_NORMALIZATION] resolved_title=%s", title)
            sources.append({
                "type": "rag",
                "title": title,
                "url": None,
            })

    # --- Web sources ---
    if include_web and web_docs:
        for doc in web_docs:
            doc_meta = doc.get("metadata", {})
            sources.append({
                "type": "web",
                "title": doc_meta.get("title") or "Web Source",
                "url": doc_meta.get("url") or None,
                "snippet": doc_meta.get("snippet") or doc.get("text", "")[:200] or None,
            })

    logger.debug("[PLANNER] sources normalized for execution_path=%s", execution_path)
    return sources


def _derive_confidence_and_disclosure(
    execution_path: str,
) -> Tuple[str, Optional[str]]:
    """Return ``(confidence_level, disclosure_note)`` for *execution_path*.

    Rules are deterministic — no heuristics, similarity scores, or doc
    counts are used.

    Returns:
        A 2-tuple of ``(confidence_level, disclosure_note)`` where
        *disclosure_note* is ``None`` when no disclaimer is needed.
    """
    if execution_path == "SYSTEM":
        confidence, disclosure = "HIGH", None
    elif "RAG" in execution_path:
        confidence, disclosure = "HIGH", None
    elif execution_path == "WEB_SEARCH":
        confidence, disclosure = (
            "MEDIUM",
            "Based on public web sources. Verify with official filings.",
        )
    elif execution_path == "LLM_ONLY":
        confidence, disclosure = (
            "LOW",
            "Generated from general knowledge. No external verification applied.",
        )
    else:
        confidence, disclosure = (
            "LOW",
            "Answer generated without verified sources.",
        )

    logger.debug("[PLANNER] confidence=%s disclosure=%s", confidence, disclosure)
    return confidence, disclosure


def _build_audit_meta(
    question: str,
    execution_path: str,
    tools_executed: List[str],
    web_docs: List[Dict],
) -> Optional[Dict[str, Any]]:
    """Build internal audit metadata when debug mode is active.

    Returns ``None`` when debug mode is off, so callers can simply omit
    the key from the response dict.
    """
    if not _is_debug_enabled():
        return None

    # Extract unique domains from web-doc URLs
    domains_used: List[str] = []
    seen: set = set()
    for doc in web_docs:
        url = doc.get("metadata", {}).get("url", "")
        if url:
            domain = urlparse(url).netloc.lower()
            if domain and domain not in seen:
                seen.add(domain)
                domains_used.append(domain)

    meta: Dict[str, Any] = {
        "tools_executed": tools_executed,
        "execution_path": execution_path,
        "temporal_override": _detect_temporal_intent(question),
        "introspection_override": _detect_system_introspection(question),
        "domains_used": domains_used,
    }

    logger.debug("[PLANNER] audit_meta=%s", meta)
    return meta


class BM25Index:
    """BM25 index using rank-bm25 library."""
    
    def __init__(self, documents: List[str], metadatas: List[Dict]):
        if not documents:
            self.index = None
            self.documents = []
            self.metadatas = []
            return
        
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.index = BM25Okapi(tokenized_docs)
        self.documents = documents
        self.metadatas = metadatas
    
    def search(self, query: str, top_k: int = 5) -> Tuple[List[str], List[Dict]]:
        if self.index is None or not self.documents:
            return [], []
        
        tokenized_query = query.lower().split()
        scores = self.index.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results_docs = [self.documents[i] for i in top_indices]
        results_metas = [self.metadatas[i] for i in top_indices]
        
        return results_docs, results_metas


def _deduplicate_documents(doc_list_1: List[str], meta_list_1: List[Dict],
                           doc_list_2: List[str], meta_list_2: List[Dict]) -> Tuple[List[str], List[Dict]]:
    """Merge and deduplicate documents from two retrieval sources."""
    seen_texts = set()
    merged_docs = []
    merged_metas = []
    
    for doc, meta in zip(doc_list_1, meta_list_1):
        doc_normalized = doc.strip().lower()
        if doc_normalized not in seen_texts:
            seen_texts.add(doc_normalized)
            merged_docs.append(doc)
            merged_metas.append(meta)
    
    for doc, meta in zip(doc_list_2, meta_list_2):
        doc_normalized = doc.strip().lower()
        if doc_normalized not in seen_texts:
            seen_texts.add(doc_normalized)
            merged_docs.append(doc)
            merged_metas.append(meta)
    
    return merged_docs, merged_metas


class LangChainOrchestrator:
    """
    LangChain-based orchestration with OpenAI-style answerability validation.
    
    Flow:
    1. ALWAYS retrieve documents first
    2. Validate answerability (entity, year, metric matching)
    3. If answerable → RAG generation
    4. If not answerable → RAG_NO_ANSWER (no LLM generation)
    5. If retrieval fails → LLM fallback
    """
    
    def __init__(self):
        """Initialize LangChain components and verify ChromaDB is not empty."""
        config = get_config()
        
        # Azure OpenAI LLM
        self.llm = AzureChatOpenAI(
            azure_endpoint=config.azure_openai.endpoint,
            azure_deployment=config.azure_openai.large_chat_deployment,
            api_key=config.azure_openai.api_key,
            api_version=config.azure_openai.api_version,
            temperature=0.0,
        )
        
        # Azure OpenAI Embeddings (MUST match ingestion model)
        # For text-embedding-3-large, don't pass model parameter (deployment name is sufficient)
        embedding_kwargs = {
            "azure_endpoint": config.azure_openai.endpoint,
            "azure_deployment": config.azure_openai.embeddings_deployment,
            "api_key": config.azure_openai.api_key,
            "api_version": config.azure_openai.api_version,
        }
        
        # Only add model parameter for older models if needed
        if "text-embedding-ada-002" in config.azure_openai.embeddings_deployment.lower():
            embedding_kwargs["model"] = "text-embedding-ada-002"
        
        self.embeddings = AzureOpenAIEmbeddings(**embedding_kwargs)
        
        self.output_parser = StrOutputParser()
        
        # Dynamic collections cache
        self.collections = {}
        self.bm25_indices = {}
        
        # Initialize web search components
        self.web_search = WebSearchEngine()
        self.investor_scraper = InvestorRelationsScraper()
        self.document_extractor = DocumentExtractor()
        self.temp_storage = get_temp_storage()
        self._current_investor_metadata = {}  # Store investor relations metadata for current query
        
        # Setup prompts and chains
        self._setup_chains()
    
    def _get_bm25_index(self, company_name: str, collection):
        """Lazy initialization of BM25 index per company."""
        from rag.company_normalizer import normalize_company_name
        norm_name = normalize_company_name(company_name)
        
        if norm_name in self.bm25_indices:
            return self.bm25_indices[norm_name]
            
        try:
            logger.info(f"Loading documents from Chroma for BM25 indexing ({norm_name})...")
            all_documents, all_metadatas = _load_all_documents_from_chroma(collection)
            
            if not all_documents:
                logger.warning(f"No documents in collection '{norm_name}'. BM25 disabled.")
                self.bm25_indices[norm_name] = None
                return None
            
            logger.info(f"Initializing BM25 index for {norm_name} with {len(all_documents)} documents...")
            index = BM25Index(all_documents, all_metadatas)
            self.bm25_indices[norm_name] = index
            return index
        except Exception as e:
            logger.error(f"Failed to setup BM25 for {norm_name}: {e}")
            return None
    
    def _setup_chains(self):
        """Setup LangChain prompts and LCEL chains."""
        
        # RAG prompt (availability-based: always answer, cite what you have)
        self.rag_template = """You are a financial analysis assistant.

Answer the question using the context documents provided below.

Context documents:
{context}

Question: {question}

Guidelines:
- Use information from the context documents to answer the question.
- Include specific numbers, dates, and facts exactly as they appear in the context.
- Preserve exact numeric values without modification.
- If the answer is explicitly present in the context, state it clearly.
- If the context only partially covers the question, answer with what is available and note which parts are not covered.
- If the context does not contain relevant information, use your general financial knowledge to provide a helpful answer and add a brief note that the response is based on general knowledge rather than internal documents.
- NEVER refuse to answer. Always provide the best possible response.
- Cite sources when referencing specific data from the context.
- For web documents, cite the official company website URL.
- For internal documents, cite the document name and page number if available.

Answer:"""
        
        self.rag_prompt = ChatPromptTemplate.from_template(self.rag_template)
        self.rag_chain = self.rag_prompt | self.llm | self.output_parser
        
        # LLM-only prompt (Azure-safe, system-perspective, no self-disclosure)
        self.llm_only_template = """You are the SageAlpha financial assistant.

Respond naturally and conversationally to the user's question.

Question: {question}

Guidelines:
- Use your general knowledge to provide helpful information.
- Respond in a clear and educational manner.
- If exact figures or current values cannot be confirmed, say "real-time data could not be verified via external sources" — NEVER say "I don't have internet access", "my training data", "my knowledge cutoff", or similar self-referential disclaimers.
- Speak from the system perspective. You ARE the system — do not describe your own limitations as an AI model.
- Provide educational information only.
- Do not give personalized or binding financial advice.

Answer:"""
        
        self.llm_only_prompt = ChatPromptTemplate.from_template(self.llm_only_template)
        self.llm_only_chain = self.llm_only_prompt | self.llm | self.output_parser
    
    # --------------------------------------------------------------------- #
    # Tuning knobs for multi-collection retrieval                            #
    # --------------------------------------------------------------------- #
    _TOP_K_PER_COLLECTION: int = 5
    _FINAL_TOP_K: int = 5

    def _retrieve_documents_hybrid(self, question: str, company_name: str) -> Tuple[List[str], List[Dict], Dict]:
        """Multi-collection retrieval: searches ALL collections in the database.

        Instead of resolving a single collection from the company name, this
        method iterates over every collection in the current Chroma database
        (``"Dev"``), performs cosine similarity search on each, aggregates the
        results, sorts globally by distance, deduplicates, and returns the
        overall top-k documents.

        Existing behaviour preserved:
        - Year-filtered retrieval when a fiscal year is detected in the query.
        - BM25 hybrid retrieval for numeric-intent queries.
        - Deduplication across all sources.
        - ``retrieval_metrics`` contract (``best_distance``, ``rag_status``, etc.).

        If ALL collections return 0 documents a warning is logged and empty
        results are returned — the caller's existing fallback logic (web
        search → LLM) handles it.
        """
        _empty_metrics: Dict = {
            "best_distance": 2.0,
            "best_similarity": 0.0,
            "rag_status": "EMPTY",
            "max_age_months": 0,
            "document_count": 0,
            "company_resolved": company_name,
            "collections_searched": 0,
            "collections_used": [],
            "avg_similarity": 0.0,
            "top_score": 0.0,
        }

        try:
            # ---------------------------------------------------------- #
            # 0. List every collection in the current database            #
            # ---------------------------------------------------------- #
            collections = list_all_collections(skip_internal=True)

            logger.info(
                "retrieval_collections_selected",
                extra={
                    "event_type": "retrieval_collections_selected",
                    "collection_count": len(collections),
                    "collections": [getattr(c, "name", str(c)) for c in collections],
                },
            )
            collections_by_name = {getattr(c, "name", str(c)): c for c in collections}

            config = get_config()
            logger.info(f"[RETRIEVER] Dataset: {config.chroma_cloud.database}")
            logger.info(
                "[RETRIEVER] Multi-collection mode: %d collections available — %s",
                len(collections),
                [getattr(c, "name", str(c)) for c in collections],
            )

            if not collections:
                logger.warning("[RETRIEVER] No searchable collections found in database")
                return [], [], _empty_metrics

            # ---------------------------------------------------------- #
            # 1. Prepare query                                            #
            # ---------------------------------------------------------- #
            requested_year = _extract_fiscal_year(question)
            if requested_year:
                logger.info(f"[RETRIEVER] Detected fiscal year in query: {requested_year}")

            query_embedding = self.embeddings.embed_query(question)
            logger.info(f"[RETRIEVER] Query embedding generated (dimension: {len(query_embedding)})")
            logger.info("[RETRIEVER] Searching ALL collections using cosine similarity")

            # ---------------------------------------------------------- #
            # 2. Search each collection (try/except per collection)       #
            # ---------------------------------------------------------- #
            # Each entry: (document_text, metadata_dict, distance, collection_name)
            all_candidates: List[Tuple[str, Dict, float, str]] = []
            collections_searched: List[str] = []
            collections_with_hits: List[str] = []

            for collection in collections:
                col_name = getattr(collection, "name", str(collection))
                try:
                    collections_searched.append(col_name)

                    # --- Year-filtered retrieval (priority) ---
                    if requested_year:
                        try:
                            yr = collection.query(
                                query_embeddings=[query_embedding],
                                n_results=self._TOP_K_PER_COLLECTION,
                                where={"fiscal_year": requested_year},
                                include=["documents", "metadatas", "distances"],
                            )
                            yr_docs = yr.get("documents", [[]])[0] if yr.get("documents") else []
                            yr_metas = yr.get("metadatas", [[]])[0] if yr.get("metadatas") else []
                            yr_dists = yr.get("distances", [[]])[0] if yr.get("distances") else []
                            for d, m, dist in zip(yr_docs, yr_metas, yr_dists):
                                m["_source_collection"] = col_name
                                m["_year_filtered"] = True
                                all_candidates.append((d, m, dist, col_name))
                            if yr_docs:
                                logger.info(
                                    "[RETRIEVER] [%s] Year-filtered: %d docs for %s",
                                    col_name, len(yr_docs), requested_year,
                                )
                        except Exception as ye:
                            logger.warning(
                                "[RETRIEVER] [%s] Year-filtered query failed: %s",
                                col_name, ye,
                            )

                    # --- General cosine retrieval ---
                    res = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=self._TOP_K_PER_COLLECTION,
                        include=["documents", "metadatas", "distances"],
                    )
                    g_docs = res.get("documents", [[]])[0] if res.get("documents") else []
                    g_metas = res.get("metadatas", [[]])[0] if res.get("metadatas") else []
                    g_dists = res.get("distances", [[]])[0] if res.get("distances") else []

                    for d, m, dist in zip(g_docs, g_metas, g_dists):
                        m["_source_collection"] = col_name
                        all_candidates.append((d, m, dist, col_name))

                    total_from_col = len(g_docs)
                    if total_from_col > 0:
                        collections_with_hits.append(col_name)
                    logger.info(
                        "[RETRIEVER] [%s] General retrieval: %d docs (best dist: %.4f)",
                        col_name,
                        total_from_col,
                        min(g_dists) if g_dists else 2.0,
                    )

                except Exception as col_err:
                    logger.warning(
                        "[RETRIEVER] Collection '%s' search failed (skipping): %s",
                        col_name, col_err,
                    )
                    continue

            # ---------------------------------------------------------- #
            # 3. Aggregate: sort globally by distance (ascending)         #
            # ---------------------------------------------------------- #
            all_candidates.sort(key=lambda x: x[2])

            logger.info(
                "[RETRIEVER] Aggregated %d candidates from %d collections",
                len(all_candidates), len(collections_searched),
            )

            # ---------------------------------------------------------- #
            # 3b. Soft metadata alignment boost (ranking only)             #
            # ---------------------------------------------------------- #
            boost_dampening_triggered = False
            boosted_count = 0
            company_boost_weight = 0.0
            year_boost_weight = 0.0
            max_total_boost = 0.0

            # Load config for metadata alignment
            try:
                cfg = get_config()
                ea_cfg = getattr(cfg, "evidence_adequacy", {}) or {}
                ma_cfg = ea_cfg.get("metadata_alignment", {}) or {}
            except Exception:
                ea_cfg = {}
                ma_cfg = {}

            boost_enabled = bool(ma_cfg.get("enabled", True))

            if boost_enabled and all_candidates:
                # Load retrieval_stability for boost dampening
                rs_cfg = ea_cfg.get("retrieval_stability", {}) or {}
                boost_dampening_threshold = float(rs_cfg.get("boost_dampening_threshold", 0.8))

                # Base weights from config
                company_boost_weight = float(ma_cfg.get("company_boost_weight", 0.08))
                year_boost_weight = float(ma_cfg.get("year_boost_weight", 0.05))
                max_total_boost = float(ma_cfg.get("max_total_boost", 0.15))

                # Normalize boost relative to distance span
                base_distances = [cand[2] for cand in all_candidates]
                min_dist = min(base_distances)
                max_dist = max(base_distances)
                distance_span = max_dist - min_dist

                span_scale = 0.5 if distance_span < 0.01 else 1.0
                eff_company_boost = company_boost_weight * span_scale
                eff_year_boost = year_boost_weight * span_scale

                company_norm: Optional[str] = None
                if company_name and company_name.strip():
                    from rag.company_normalizer import normalize_company_name
                    company_norm = normalize_company_name(company_name)

                # First pass: compute raw boost per candidate and track per-collection boosted counts
                raw_boosts: List[Tuple[str, Dict, float, float, str]] = []
                boosted_per_collection: Dict[str, int] = {}
                for doc, meta, dist, cname in all_candidates:
                    base_dist = dist
                    boost = 0.0

                    # Company match boost
                    if company_norm:
                        meta_company = str(meta.get("company") or meta.get("_source_collection") or "").strip()
                        if meta_company:
                            try:
                                if normalize_company_name(meta_company) == company_norm:
                                    boost += eff_company_boost
                            except Exception:
                                pass

                    # Fiscal year match boost
                    fiscal_year = str(meta.get("fiscal_year") or "").strip()
                    if requested_year and fiscal_year:
                        try:
                            fy_norm = str(fiscal_year).replace("FY", "").strip()
                            rq_norm = str(requested_year).replace("FY", "").strip()
                            if fy_norm == rq_norm:
                                boost += eff_year_boost
                        except Exception:
                            pass

                    # Cap total boost
                    if max_total_boost > 0.0:
                        boost = min(boost, max_total_boost)

                    if boost > 0.0:
                        boosted_count += 1
                        boosted_per_collection[cname] = boosted_per_collection.get(cname, 0) + 1

                    raw_boosts.append((doc, meta, base_dist, boost, cname))

                # Boost concentration guard: dampen collections with > threshold share of boosted docs
                total_boosted_docs = boosted_count
                dampened_collections: Dict[str, bool] = {}
                if total_boosted_docs > 0:
                    for col, count in boosted_per_collection.items():
                        if count / total_boosted_docs > boost_dampening_threshold:
                            dampened_collections[col] = True
                boost_dampening_triggered = bool(dampened_collections)

                boosted_candidates = []
                col_boost_sums: Dict[str, List] = {c: [0.0, 0.0, 0] for c in dampened_collections}
                for doc, meta, base_dist, boost, cname in raw_boosts:
                    if cname in dampened_collections:
                        effective_boost = boost * 0.5
                    else:
                        effective_boost = boost
                    if max_total_boost > 0.0:
                        effective_boost = min(effective_boost, max_total_boost)
                    adjusted_dist = max(0.0, base_dist - effective_boost)
                    boosted_candidates.append((doc, meta, adjusted_dist, cname))
                    if cname in col_boost_sums and boost > 0.0:
                        col_boost_sums[cname][0] += boost
                        col_boost_sums[cname][1] += effective_boost
                        col_boost_sums[cname][2] += 1

                for col in dampened_collections:
                    s = col_boost_sums[col]
                    n = s[2]
                    if n > 0:
                        orig = s[0] / n
                        reduced = s[1] / n
                        logger.debug(
                            "boost_dampening_applied",
                            extra={
                                "event_type": "boost_dampening_applied",
                                "collection": col,
                                "original_boost": orig,
                                "reduced_boost": reduced,
                            },
                        )

                boosted_candidates.sort(key=lambda x: x[2])
                all_candidates = boosted_candidates

            logger.debug(
                "metadata_alignment_applied",
                extra={
                    "event_type": "metadata_alignment",
                    "boost_enabled": boost_enabled,
                    "company_boost_weight": company_boost_weight,
                    "year_boost_weight": year_boost_weight,
                    "max_total_boost": max_total_boost,
                    "boosted_docs": boosted_count,
                },
            )

            # ---------------------------------------------------------- #
            # 3c. Diversity floor (soft rebalance, no filtering)           #
            # ---------------------------------------------------------- #
            try:
                rs_cfg = get_config()
                rs_cfg = getattr(rs_cfg, "evidence_adequacy", {}) or {}
                rs_cfg = rs_cfg.get("retrieval_stability", {}) or {}
            except Exception:
                rs_cfg = {}
            stability_enabled = bool(rs_cfg.get("enabled", True))
            dominance_threshold = float(rs_cfg.get("dominance_threshold", 0.7))
            dominance_penalty = float(rs_cfg.get("dominance_penalty", 0.02))

            if stability_enabled and all_candidates:
                window_size = min(len(all_candidates), 2 * self._FINAL_TOP_K)
                window = all_candidates[:window_size]
                col_counts: Dict[str, int] = {}
                for _d, _m, _dist, cname in window:
                    col_counts[cname] = col_counts.get(cname, 0) + 1
                total_in_window = len(window)
                dominant_collection: Optional[str] = None
                dominance_ratio = 0.0
                for cname, count in col_counts.items():
                    ratio = count / total_in_window if total_in_window else 0.0
                    if ratio > dominance_ratio:
                        dominance_ratio = ratio
                        dominant_collection = cname
                penalty_applied = (
                    dominant_collection is not None and dominance_ratio > dominance_threshold
                )
                if penalty_applied and dominant_collection is not None:
                    rebalanced: List[Tuple[str, Dict, float, str]] = []
                    for idx, (doc, meta, dist, cname) in enumerate(all_candidates):
                        if idx < window_size and cname == dominant_collection:
                            new_dist = dist + dominance_penalty
                        else:
                            new_dist = dist
                        rebalanced.append((doc, meta, new_dist, cname))
                    rebalanced.sort(key=lambda x: x[2])
                    all_candidates = rebalanced
                logger.debug(
                    "diversity_rebalance",
                    extra={
                        "event_type": "diversity_rebalance",
                        "dominant_collection": dominant_collection or "",
                        "dominance_ratio": dominance_ratio,
                        "penalty_applied": penalty_applied,
                    },
                )

            # ---------------------------------------------------------- #
            # 3d. Retrieval stability telemetry (before dedupe)            #
            # ---------------------------------------------------------- #
            if stability_enabled and all_candidates:
                top5 = all_candidates[: min(5, len(all_candidates))]
                top5_collections = [c[3] for c in top5]
                top5_unique_collections = len(set(top5_collections))
                col_counts_top5: Dict[str, int] = {}
                for cname in top5_collections:
                    col_counts_top5[cname] = col_counts_top5.get(cname, 0) + 1
                n5 = len(top5)
                dominance_ratio_top5 = (
                    max(col_counts_top5.values()) / n5 if n5 else 0.0
                )
                entropy = 0.0
                if n5 > 0:
                    for c in col_counts_top5.values():
                        p = c / n5
                        if p > 0.0:
                            entropy -= p * math.log(p)
                logger.debug(
                    "retrieval_stability",
                    extra={
                        "event_type": "retrieval_stability",
                        "top5_unique_collections": top5_unique_collections,
                        "dominance_ratio": dominance_ratio_top5,
                        "entropy": entropy,
                    },
                )

            # ---------------------------------------------------------- #
            # 4. Deduplicate by normalised text                           #
            # ---------------------------------------------------------- #
            seen_texts: set = set()
            deduped: List[Tuple[str, Dict, float, str]] = []
            for doc, meta, dist, cname in all_candidates:
                key = doc.strip().lower()
                if key not in seen_texts:
                    seen_texts.add(key)
                    deduped.append((doc, meta, dist, cname))

            # ---------------------------------------------------------- #
            # 5. Select final top_k                                       #
            # ---------------------------------------------------------- #
            top_results = deduped[: self._FINAL_TOP_K]

            chroma_docs = [r[0] for r in top_results]
            chroma_metas = [r[1] for r in top_results]
            chroma_distances = [r[2] for r in top_results]

            if chroma_distances:
                logger.info(f"[RETRIEVER] Top distances after global sort: {chroma_distances}")
                logger.info(f"[RETRIEVER] Best match distance: {chroma_distances[0]:.4f}")

            if chroma_metas:
                sample = chroma_metas[0]
                logger.info(
                    "[RETRIEVER] Best match → collection=%s company=%s year=%s",
                    sample.get("_source_collection", "N/A"),
                    sample.get("company", "N/A"),
                    sample.get("fiscal_year", "N/A"),
                )

            if not chroma_docs:
                logger.warning(
                    "[RETRIEVER] ALL %d collections returned 0 documents — "
                    "web fallback will be triggered by caller",
                    len(collections_searched),
                )

            # ---------------------------------------------------------- #
            # 6. BM25 hybrid (numeric intent) — on best-hit collection    #
            # ---------------------------------------------------------- #
            bm25_docs: List[str] = []
            bm25_metas: List[Dict] = []
            if _detect_numeric_intent(question) and collections_with_hits:
                # Run BM25 on the collection that contributed the best hit
                best_col_name = top_results[0][3] if top_results else None
                if best_col_name:
                    best_collection = collections_by_name.get(best_col_name) or get_company_collection(best_col_name)
                    try:
                        doc_count = best_collection.count()
                        if doc_count == 0:
                            logger.debug("bm25_skipped_empty_collection")
                        else:
                            bm25_index = self._get_bm25_index(best_col_name, best_collection)
                            if bm25_index is not None:
                                bm25_docs, bm25_metas = bm25_index.search(question, top_k=5)
                                logger.info(f"[RETRIEVER] BM25 retrieved {len(bm25_docs)} docs from '{best_col_name}'")
                    except Exception as bm25_err:
                        logger.warning(f"[RETRIEVER] BM25 failed on '{best_col_name}': {bm25_err}")

            # ---------------------------------------------------------- #
            # 7. Final merge and deduplicate                              #
            # ---------------------------------------------------------- #
            merged_docs, merged_metas = _deduplicate_documents(
                chroma_docs, chroma_metas,
                bm25_docs, bm25_metas,
            )

            # ---------------------------------------------------------- #
            # 8. Calculate metrics                                        #
            # ---------------------------------------------------------- #
            distances = list(chroma_distances) if chroma_distances else []
            best_dist = distances[0] if distances else 2.0
            second_dist = distances[1] if len(distances) > 1 else best_dist
            distance_gap = second_dist - best_dist
            distance_variance = (
                statistics.pvariance(distances) if len(distances) > 1 else 0.0
            )

            distinct_docs = len(set(
                (meta.get("source") or meta.get("file_name") or "")
                for meta in merged_metas
            ))
            diversity_ratio = distinct_docs / max(1, len(merged_metas))

            n_metas = max(1, len(merged_metas))
            if requested_year:
                year_matches = sum(
                    1 for m in merged_metas
                    if (m.get("fiscal_year") or "").strip() == requested_year.strip()
                    or str(m.get("fiscal_year") or "").replace("FY", "").strip() == requested_year.replace("FY", "").strip()
                )
                year_match_ratio = year_matches / n_metas
            else:
                year_match_ratio = 1.0

            if company_name and company_name.strip():
                from rag.company_normalizer import normalize_company_name
                company_norm = normalize_company_name(company_name)
                company_matches = sum(
                    1 for m in merged_metas
                    if normalize_company_name(str(m.get("company") or m.get("_source_collection") or "")) == company_norm
                )
                company_match_ratio = company_matches / n_metas
            else:
                company_match_ratio = 1.0

            avg_dist = (
                sum(chroma_distances) / len(chroma_distances)
                if chroma_distances
                else 2.0
            )
            best_sim = max(0.0, 1.0 - (best_dist / 2.0))
            avg_sim = max(0.0, 1.0 - (avg_dist / 2.0))

            # --- STABILIZATION: Retrieval Quality Threshold (Rule 3) ---
            rag_status = "FOUND" if merged_docs else "EMPTY"
            if best_dist >= 1.5 and rag_status == "FOUND":
                logger.warning(f"[RETRIEVER] LOW_CONFIDENCE match detected (dist={best_dist:.4f} >= 1.5)")
                rag_status = "LOW_CONFIDENCE"

            retrieval_metrics: Dict = {
                "best_distance": best_dist,
                "best_similarity": best_sim,
                "avg_similarity": avg_sim,
                "top_score": best_sim,
                "rag_status": rag_status,
                "max_age_months": 0,
                "document_count": len(merged_docs),
                "company_resolved": company_name,
                "collections_searched": len(collections_searched),
                "collections_used": list(set(
                    r[3] for r in top_results
                )),
                "distance_gap": distance_gap,
                "distance_variance": distance_variance,
                "diversity_ratio": diversity_ratio,
                "year_match_ratio": year_match_ratio,
                "company_match_ratio": company_match_ratio,
            }

            # ---------------------------------------------------------- #
            # 8b. Retrieval trust score (telemetry + adequacy signal)     #
            # ---------------------------------------------------------- #
            try:
                _cfg = get_config()
                _ea = getattr(_cfg, "evidence_adequacy", {}) or {}
                _rt = _ea.get("retrieval_trust", {}) or {}
            except Exception:
                _rt = {}
            trust_enabled = bool(_rt.get("enabled", True))
            distance_gap_threshold = float(_rt.get("distance_gap_threshold", 0.25))
            min_entropy_normalizer = float(_rt.get("min_entropy_normalizer", 1.0))
            if min_entropy_normalizer <= 0.0:
                min_entropy_normalizer = math.log(5.0)

            diversity_factor = 1.0
            entropy_factor = 1.0
            coherence_penalty = 1.0
            dominance_factor = 1.0
            if trust_enabled and top_results:
                top5 = top_results[: min(5, len(top_results))]
                top5_unique = len(set(r[3] for r in top5))
                diversity_factor = min(1.0, top5_unique / 3.0)
                col_counts_t5: Dict[str, int] = {}
                for _d, _m, _dist, cname in top5:
                    col_counts_t5[cname] = col_counts_t5.get(cname, 0) + 1
                n5 = len(top5)
                entropy_trust = 0.0
                if n5 > 0:
                    for c in col_counts_t5.values():
                        p = c / n5
                        if p > 0.0:
                            entropy_trust -= p * math.log(p)
                entropy_factor = min(1.0, entropy_trust / min_entropy_normalizer)
                coherence_penalty = 0.7 if distance_gap > distance_gap_threshold else 1.0
                dominance_factor = 0.8 if boost_dampening_triggered else 1.0
                trust_score = (
                    diversity_factor
                    * entropy_factor
                    * coherence_penalty
                    * dominance_factor
                )
                trust_score = max(0.0, min(1.0, trust_score))
            else:
                trust_score = 1.0

            retrieval_metrics["trust_score"] = trust_score
            logger.debug(
                "retrieval_trust_evaluated",
                extra={
                    "event_type": "retrieval_trust_evaluated",
                    "trust_score": trust_score,
                    "diversity_factor": diversity_factor,
                    "entropy_factor": entropy_factor,
                    "coherence_penalty": coherence_penalty,
                    "dominance_factor": dominance_factor,
                },
            )

            # Estimate age from metadata
            from datetime import datetime
            current_year = datetime.now().year
            ages_months: List[int] = []
            for m in merged_metas:
                fy = m.get("fiscal_year")
                if fy and str(fy).startswith("FY"):
                    try:
                        fy_year = int(str(fy)[2:])
                        ages_months.append((current_year - fy_year) * 12)
                    except Exception:
                        pass
                elif fy and str(fy).isdigit():
                    ages_months.append((current_year - int(fy)) * 12)

            if ages_months:
                retrieval_metrics["max_age_months"] = max(ages_months)

            logger.info(f"[RETRIEVER] Total documents retrieved: {len(merged_docs)}")
            logger.info(
                "[RETRIEVER] Collections used: %s", retrieval_metrics["collections_used"],
            )

            # Log first document preview if available
            if merged_docs:
                first_doc_preview = merged_docs[0][:500]
                logger.info(f"[RETRIEVER] First document preview: {first_doc_preview}...")

            return merged_docs, merged_metas, retrieval_metrics

        except Exception as e:
            logger.error(f"[RETRIEVER] Multi-collection retrieval failed: {e}", exc_info=True)
            return [], [], _empty_metrics
    
    def _retrieve_investor_relations_evidence(
        self,
        question: str,
        company_name: str,
    ) -> List[Dict]:
        """Investor-relations scraping sub-flow for ``_retrieve_web_evidence``.

        Searches for the company's investor-relations page, scrapes it, and
        downloads / extracts priority financial documents (annual reports,
        10-K, earnings releases, etc.).

        Returns:
            A list of web-document dicts (same format as
            ``_retrieve_web_evidence``), or an empty list when the flow
            cannot complete (page not found, scrape failure, etc.).
        """
        search_result = self.web_search.search_company_investor_relations(company_name)

        if not search_result.get("success"):
            logger.warning(
                "[WEB_SEARCH] Could not find investor relations page: %s",
                search_result.get("error"),
            )
            return []

        investor_url = search_result.get("investor_url")
        official_domain = search_result.get("official_domain")
        if not investor_url:
            logger.warning("[WEB_SEARCH] Investor URL not found")
            return []

        # Store investor relations metadata for source formatting
        self._current_investor_metadata = {
            "investor_url": investor_url,
            "official_domain": official_domain,
            "company": company_name,
        }

        # Scrape investor page
        logger.info(f"[WEB_SEARCH] Scraping investor page: {investor_url}")
        scrape_result = self.investor_scraper.scrape_investor_page(investor_url)

        if not scrape_result.get("success"):
            logger.warning(
                "[WEB_SEARCH] Failed to scrape investor page: %s",
                scrape_result.get("error"),
            )
            return []

        # Download and extract documents
        web_documents: List[Dict] = []
        documents_found = scrape_result.get("documents", [])

        # Prioritize annual reports and financial statements
        priority_types = [
            "annual_report", "financial_statement", "10-k", "10-q",
            "earnings_release",
        ]
        priority_docs = [
            d for d in documents_found if d.get("type") in priority_types
        ]
        other_docs = [
            d for d in documents_found if d.get("type") not in priority_types
        ]

        # Process priority documents first (limit to 3 to avoid too many downloads)
        for doc_info in (priority_docs + other_docs)[:3]:
            doc_url = doc_info.get("url")
            if not doc_url:
                continue

            try:
                logger.info(
                    "[WEB_SEARCH] Downloading document: %s",
                    doc_info.get("title", "Unknown"),
                )
                file_path, download_meta = self.investor_scraper.download_document(doc_url)

                if not file_path:
                    continue

                # Extract text
                extract_result = self.document_extractor.extract_text(
                    file_path, download_meta
                )

                if extract_result.get("success"):
                    # Store in temporary storage with investor relations metadata
                    storage_meta = self.temp_storage.store_document(
                        file_path,
                        {
                            **download_meta,
                            "title": doc_info.get("title", ""),
                            "type": doc_info.get("type", ""),
                            "url": doc_url,
                            "investor_url": self._current_investor_metadata.get(
                                "investor_url"
                            ),
                            "investor_root": self._current_investor_metadata.get(
                                "investor_url"
                            ),
                            "official_domain": self._current_investor_metadata.get(
                                "official_domain"
                            ),
                            "company": self._current_investor_metadata.get(
                                "company"
                            ),
                        },
                    )

                    web_documents.append({
                        "text": extract_result.get("text", ""),
                        "metadata": {
                            **storage_meta,
                            "title": doc_info.get("title", ""),
                            "url": doc_url,
                            "type": doc_info.get("type", ""),
                        },
                    })

                    logger.info(
                        "[WEB_SEARCH] Extracted %d pages from document",
                        extract_result.get("pages", 0),
                    )

            except Exception as e:
                logger.error(
                    "[WEB_SEARCH] Failed to process document %s: %s",
                    doc_url, e,
                )
                continue

        logger.info(
            "[WEB_SEARCH] Investor-relations path retrieved %d web documents",
            len(web_documents),
        )
        return web_documents

    def _retrieve_web_evidence(
        self,
        question: str,
        company_name: Optional[str] = None
    ) -> List[Dict]:
        """
        Retrieve evidence from web search (company investor relations).
        
        Returns:
            List of document dicts with text and metadata
        """
        if not self.web_search.enabled:
            logger.info("[WEB_SEARCH] Web search disabled, skipping")
            return []
        
        try:
            if not company_name:
                company_name = extract_company_name(question)

            if company_name:
                logger.info(f"[WEB_SEARCH] Searching for company: {company_name}")
            else:
                logger.info(
                    "[WEB_SEARCH] No company detected — proceeding with query-only web search"
                )

            # ---------------------------------------------------------- #
            # Company-specific path: investor relations scraping          #
            # ---------------------------------------------------------- #
            if company_name:
                ir_docs = self._retrieve_investor_relations_evidence(
                    question, company_name
                )
                if ir_docs:
                    return ir_docs
                # Company IR search unsuccessful — fall through to
                # query-only search below so the question still gets
                # answered from priority / fallback domains.
                logger.info(
                    "[WEB_SEARCH] Company investor-relations search returned 0 "
                    "documents — falling back to query-only search"
                )

            # ---------------------------------------------------------- #
            # Query-only path: phased search using the full question      #
            # ---------------------------------------------------------- #
            search_query = (
                f"{company_name} {question}" if company_name else question
            )
            raw_results = self.web_search._phased_search(search_query, num=10)

            if not raw_results:
                logger.info("[WEB_SEARCH] Query-only phased search returned 0 results")
                return []

            # Convert search results to the standard web-document format
            web_documents: List[Dict] = []
            for result in raw_results[:5]:
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                link = result.get("link", "")
                if snippet:
                    web_documents.append({
                        "text": f"{title}\n\n{snippet}",
                        "metadata": {
                            "title": title,
                            "url": link,
                            "type": "web_search_result",
                            "source": "web_search",
                        },
                    })

            logger.info(
                "[WEB_SEARCH] Retrieved %d web documents via query-only search",
                len(web_documents),
            )
            return web_documents
            
        except Exception as e:
            logger.error(f"[WEB_SEARCH] Web evidence retrieval failed: {e}", exc_info=True)
            return []
    
    def answer_query(self, question: str) -> Dict[str, Any]:
        """
        Availability-based query routing.

        Routing is determined by document availability, NOT company identity.

        Flow:
            1. ALWAYS attempt RAG retrieval for every query.
            2. If RAG returns 0 documents or low similarity → trigger Web Search.
            3. If Web Search returns documents → WEB_SEARCH + LLM.
            4. If neither RAG nor Web Search returns documents → LLM-only
               with a soft disclaimer (NEVER hard-block).

        Resulting paths:
            - RAG + LLM           → ``sagealpha_rag``
            - RAG + WEB + LLM     → ``sagealpha_hybrid_search``
            - WEB_SEARCH + LLM    → ``sagealpha_ai_search``
            - LLM_ONLY            → ``sagealpha_rag`` (with disclaimer)
        """
        # Safe defaults
        answer = ""
        answer_type = "sagealpha_rag"
        formatted_sources: List[Any] = []
        rag_chain_executed = False
        rag_chain_error = None

        try:
            logger.info("=" * 60)
            logger.info("[QUERY] Processing query (availability-based routing)")
            logger.info("=" * 60)
            logger.info("[QUERY] Query text: %s", question[:120])

            # -------------------------------------------------------------- #
            # STEP 1: ALWAYS attempt RAG retrieval                            #
            # -------------------------------------------------------------- #
            # Resolve company for dynamic collection (Rule 1, 4, 7)
            try:
                company_name = extract_company_name(question)
                if not company_name:
                    raise ValueError("Company name could not be resolved.")
            except Exception as e:
                logger.error(f"[ROUTER] Extraction failed: {e}")
                return {
                    "answer": "Company name could not be resolved. Please specify a company clearly.",
                    "answer_type": "ERROR",
                    "sources": [],
                    "confidence_level": "LOW",
                }

            logger.info("[STEP 1] Attempting RAG retrieval for ALL queries")
            documents, metadatas, retrieval_metrics = self._retrieve_documents_hybrid(question, company_name)
            logger.info("[STEP 1] RAG retrieved %d documents", len(documents))

            # -------------------------------------------------------------- #
            # STEP 2: Adequacy-based web escalation (no distance gating)     #
            # -------------------------------------------------------------- #
            web_documents: List[Dict] = []

            if len(documents) == 0:
                logger.info("[STEP 2] RAG returned 0 documents → triggering WEB_SEARCH")
                web_documents = self._retrieve_web_evidence(question, company_name)
                logger.info("[STEP 2] WEB_SEARCH retrieved %d documents", len(web_documents))
            else:
                from rag.evidence_adequacy import EvidenceAdequacyEvaluator
                adequacy = EvidenceAdequacyEvaluator.evaluate(
                    question=question,
                    rag_docs=documents,
                    rag_metas=metadatas,
                    best_distance=retrieval_metrics.get("best_distance", 1.5),
                    llm=self.llm,
                    retrieval_metrics=retrieval_metrics,
                )
                logger.info(
                    "confidence_gating_decision",
                    extra={
                        "event_type": "confidence_gating",
                        "decision": adequacy["decision"],
                        "final_score": adequacy["final_score"],
                        "llm_score": adequacy["llm_score"],
                        "retrieval_score": adequacy["retrieval_score"],
                        "doc_type_score": adequacy["doc_type_score"],
                    },
                )
                if adequacy["decision"] == "ESCALATE_WEB":
                    logger.info("[STEP 2] Adequacy ESCALATE_WEB → triggering WEB_SEARCH")
                    web_documents = self._retrieve_web_evidence(question, company_name)
                    logger.info("[STEP 2] WEB_SEARCH retrieved %d documents", len(web_documents))
                elif adequacy["decision"] == "LLM_ONLY":
                    logger.info("[STEP 2] Adequacy LLM_ONLY → discarding RAG docs")
                    documents = []
                    metadatas = []

            # -------------------------------------------------------------- #
            # STEP 3: Determine execution path                                #
            # -------------------------------------------------------------- #
            has_rag = len(documents) > 0
            has_web = len(web_documents) > 0

            if has_rag and has_web:
                execution_path = "RAG+WEB_SEARCH+LLM"
            elif has_rag:
                execution_path = "RAG+LLM"
            elif has_web:
                execution_path = "WEB_SEARCH+LLM"
            else:
                execution_path = "LLM_ONLY"

            logger.info("=" * 60)
            logger.info("[ROUTER] execution_path=%s", execution_path)
            logger.info(
                "[ROUTER] rag_docs=%d  web_docs=%d",
                len(documents),
                len(web_documents),
            )
            logger.info("=" * 60)

            # -------------------------------------------------------------- #
            # ANSWERABILITY GATE                                               #
            # -------------------------------------------------------------- #
            _gate_rag = documents if has_rag else []
            _gate_metas = metadatas if has_rag else []
            answerable, gate_reason = _is_answerable(
                question, _gate_rag, _gate_metas, web_documents,
            )
            if not answerable:
                logger.warning(
                    "[GUARDRAIL] Answerability gate BLOCKED answer: %s",
                    gate_reason,
                )
                MetricsRecorder.record_blocked_query(
                    question, gate_reason, has_documents=has_rag or has_web,
                )
                return {
                    "answer": (
                        "No verified documents matching the requested "
                        "company, time period, and metrics were found. "
                        "Exact figures cannot be provided."
                    ),
                    "answer_type": "NO_ANSWER",
                    "sources": [],
                    "confidence_level": "LOW",
                    "disclosure_note": gate_reason,
                }

            # -------------------------------------------------------------- #
            # STEP 4: Generate answer                                         #
            # -------------------------------------------------------------- #
            if has_rag or has_web:
                # We have evidence — fuse and answer via RAG chain
                rag_docs_for_fusion = documents if has_rag else []
                rag_metas_for_fusion = metadatas if has_rag else []

                fusion_result = EvidenceFusion.fuse_evidence(
                    rag_docs_for_fusion,
                    rag_metas_for_fusion,
                    web_documents,
                    question,
                )
                fused_context = fusion_result.get("fused_context", "")

                try:
                    set_llm_cost_stage("answer_generation")
                    if fused_context:
                        answer = self.rag_chain.invoke(
                            {"question": question, "context": fused_context},
                            config={"callbacks": [cost_tracker_callback]},
                        )
                    else:
                        # Fallback: assemble context manually
                        context_parts: List[str] = []
                        for doc, meta in zip(rag_docs_for_fusion, rag_metas_for_fusion):
                            meta_info = ""
                            if meta.get("source"):
                                meta_info = f"Source: {meta['source']}"
                            if meta.get("fiscal_year"):
                                meta_info += f", FY: {meta['fiscal_year']}"
                            if meta.get("page"):
                                meta_info += f", Page: {meta['page']}"
                            context_parts.append(
                                f"[{meta_info}]\n{doc}" if meta_info else doc
                            )
                        answer = self.rag_chain.invoke(
                            {
                                "question": question,
                                "context": "\n\n---\n\n".join(context_parts),
                            },
                            config={"callbacks": [cost_tracker_callback]},
                        )
                    rag_chain_executed = True
                    logger.info("[STEP 4] LLM answer generated from evidence")

                except Exception as rag_error:
                    rag_chain_error = rag_error
                    logger.error(
                        "[STEP 4] RAG chain failed (%s: %s) — falling back to LLM-only",
                        type(rag_error).__name__,
                        str(rag_error)[:200],
                    )
                    try:
                        set_llm_cost_stage("answer_generation")
                        answer = self.llm_only_chain.invoke(
                            {"question": question},
                            config={"callbacks": [cost_tracker_callback]},
                        )
                        logger.warning("[FALLBACK] LLM-only answer generated (RAG chain error)")
                    except Exception as llm_error:
                        logger.error("[CRITICAL] LLM fallback also failed: %s", llm_error)
                        raise RuntimeError(
                            f"Both RAG and LLM generation failed. "
                            f"RAG error: {rag_error}, LLM error: {llm_error}"
                        ) from llm_error
            else:
                # ----- LLM_ONLY path with soft disclaimer ----- #
                logger.info("[STEP 4] No evidence available — using LLM-only with disclaimer")
                MetricsRecorder.record_llm_fallback(question, reason="no_evidence")
                set_llm_cost_stage("answer_generation")
                answer = self.llm_only_chain.invoke(
                    {"question": question},
                    config={"callbacks": [cost_tracker_callback]},
                )

                # Append disclaimer — stronger for numeric/financial queries
                if _detect_numeric_intent(question):
                    _DISCLAIMER = (
                        "\n\n---\n⚠️ **Note:** This answer is based on general "
                        "knowledge and may not reflect audited or finalized "
                        "financial figures. For authoritative values, please "
                        "consult the company's official investor relations "
                        "filings or regulatory disclosures."
                    )
                    logger.info("[STEP 4] LLM-only answer generated with FINANCIAL disclaimer")
                else:
                    _DISCLAIMER = (
                        "\n\n---\n*Note: This answer is based on the model's general "
                        "knowledge. For verified financial data, please refer to the "
                        "company's official investor relations website or annual reports.*"
                    )
                    logger.info("[STEP 4] LLM-only answer generated with soft disclaimer")
                answer = answer.rstrip() + _DISCLAIMER

            # -------------------------------------------------------------- #
            # STEP 5: Determine answer_type                                   #
            # -------------------------------------------------------------- #
            if rag_chain_executed:
                if has_rag and has_web:
                    answer_type = "sagealpha_hybrid_search"
                elif has_web:
                    answer_type = "sagealpha_ai_search"
                else:
                    answer_type = "sagealpha_rag"
            elif has_rag or has_web:
                # Chain failed but we had evidence — label based on sources
                if has_rag and has_web:
                    answer_type = "sagealpha_hybrid_search"
                elif has_web:
                    answer_type = "sagealpha_ai_search"
                else:
                    answer_type = "sagealpha_rag"
                logger.warning(
                    "[STEP 5] answer_type=%s (chain failed, LLM fallback)", answer_type
                )
            else:
                answer_type = "sagealpha_rag"

            # -------------------------------------------------------------- #
            # STEP 6: Normalize sources                                       #
            # -------------------------------------------------------------- #
            source_metas = metadatas if has_rag else []
            formatted_sources = _normalize_sources(
                source_metas, web_documents, execution_path
            )

            # -------------------------------------------------------------- #
            # STEP 7: Summary log + return                                    #
            # -------------------------------------------------------------- #
            logger.info("=" * 60)
            logger.info("[RESPONSE] ROUTING SUMMARY")
            logger.info("=" * 60)
            logger.info("[RESPONSE]   execution_path = %s", execution_path)
            logger.info("[RESPONSE]   answer_type    = %s", answer_type)
            logger.info("[RESPONSE]   rag_docs       = %d", len(documents))
            logger.info("[RESPONSE]   web_docs       = %d", len(web_documents))
            logger.info(
                "[RESPONSE]   sources_count  = %d", len(formatted_sources),
            )
            logger.info("=" * 60)

            # -------------------------------------------------------------- #
            # STEP 8: Post-Processing & Validation                           #
            # -------------------------------------------------------------- #
            # 1. Confidence Scoring
            confidence_score = ConfidenceScorer.calculate_score(
                execution_path=execution_path,
                rag_metas=source_metas,
                web_docs=web_documents
            )
            confidence_level = ConfidenceScorer.get_confidence_level(confidence_score)
            
            # Get disclosure note from legacy logic
            _, disclosure_note = _derive_confidence_and_disclosure(execution_path)

            # 2. Numeric Validation Guardrail
            all_raw_docs = [doc for doc in documents] + [d.get("text", "") for d in web_documents]
            answer = NumericValidator.apply_guardrail(answer, all_raw_docs)

            # Derive tools_executed for audit
            _legacy_tools: List[str] = []
            if has_rag:
                _legacy_tools.append("RAG")
            if has_web:
                _legacy_tools.append("WEB_SEARCH")
            _legacy_tools.append("LLM")
            audit_meta = _build_audit_meta(
                question, execution_path, _legacy_tools, web_documents
            )

            result: Dict[str, Any] = {
                "answer": answer,
                "answer_type": answer_type,
                "sources": formatted_sources,
                "confidence_level": confidence_level,
                "confidence_score": confidence_score,
                "disclosure_note": disclosure_note,
            }
            if audit_meta is not None:
                result["_meta"] = audit_meta
            if rag_chain_executed:
                MetricsRecorder.record_rag_answer(
                    question, has_documents=has_rag or has_web,
                )
            return result

        except Exception as e:
            logger.error("LangChain orchestration failed: %s", e, exc_info=True)
            return {
                "answer": (
                    "I apologize, but I encountered an error while processing "
                    "your query. Please try again."
                ),
                "answer_type": "sagealpha_rag",
                "sources": [],
            }
    


# Singleton instance
_orchestrator: Optional[LangChainOrchestrator] = None
_orchestrator_error: Optional[str] = None


def get_llm_only_chain():
    """
    Get lightweight LLM-only chain without initializing RAG components.
    Use this for greetings and general knowledge queries.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from config.settings import get_config
    
    config = get_config()
    
    # Create LLM instance (no RAG components)
    llm = AzureChatOpenAI(
        azure_endpoint=config.azure_openai.endpoint,
        azure_deployment=config.azure_openai.large_chat_deployment,
        api_key=config.azure_openai.api_key,
        api_version=config.azure_openai.api_version,
        temperature=0.0,
    )
    
    # LLM-only prompt (Azure-safe, system-perspective, no self-disclosure)
    llm_only_template = """You are the SageAlpha financial assistant.

Answer the user's question using your general knowledge.
Keep responses clear and helpful.

If exact or current figures cannot be confirmed, say "real-time data could not be verified via external sources" — NEVER say "I don't have internet access", "my training data", "my knowledge cutoff", or similar self-referential disclaimers.
Speak from the system perspective. You ARE the system — do not describe your own limitations as an AI model.
Provide educational information only and avoid personalized financial advice.

Question: {question}

Answer:"""
    
    llm_only_prompt = ChatPromptTemplate.from_template(llm_only_template)
    llm_only_chain = llm_only_prompt | llm | StrOutputParser()
    
    return llm_only_chain


def get_orchestrator() -> LangChainOrchestrator:
    """Get singleton orchestrator instance."""
    global _orchestrator, _orchestrator_error
    
    if _orchestrator is not None:
        return _orchestrator
    
    if _orchestrator_error is not None:
        raise RuntimeError(f"Orchestrator initialization failed previously: {_orchestrator_error}")
    
    try:
        logger.info("[ORCHESTRATOR] Initializing LangChain orchestrator...")
        _orchestrator = LangChainOrchestrator()
        logger.info("[ORCHESTRATOR] Orchestrator initialized successfully")
        return _orchestrator
    except RuntimeError as e:
        # ChromaDB empty or fatal errors
        error_msg = str(e)
        _orchestrator_error = error_msg
        logger.error(f"[ORCHESTRATOR] Initialization failed: {error_msg}")
        raise
    except ValueError as e:
        # Configuration errors
        error_msg = str(e)
        _orchestrator_error = error_msg
        logger.error(f"[ORCHESTRATOR] Configuration error: {error_msg}")
        raise
    except Exception as e:
        # Other initialization errors
        error_msg = str(e)
        _orchestrator_error = error_msg
        logger.error(f"[ORCHESTRATOR] Unexpected initialization error: {error_msg}", exc_info=True)
        raise RuntimeError(f"Failed to initialize orchestrator: {error_msg}") from e


# ---------------------------------------------------------------------------
# LLM-based company name extraction
# ---------------------------------------------------------------------------


    # extract_company_name_llm moved to rag/company_extractor.py


def _execute_planner_plan(
    orchestrator: LangChainOrchestrator,
    question: str,
    plan: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute a planner-produced execution plan using existing tool logic.

    Iterates over the ordered ``execution_plan`` steps and delegates to
    the orchestrator's existing retrieval / generation methods.  No intent
    classification, company gates, or heuristic thresholds are applied.

    Safety behaviours:
        * **RAG → WEB_SEARCH auto-escalation** — if a RAG step returns zero
          documents *and* no WEB_SEARCH step is already planned later in the
          execution plan, a WEB_SEARCH retrieval is automatically injected.
        * **Structured observability** — every retrieval result count,
          escalation decision, and the final execution path are logged.
        * The response includes an internal ``_planner_meta`` key (popped
          by the caller before returning to the API) so that
          ``answer_query_simple`` can decide whether to re-plan.

    Args:
        orchestrator: Initialised orchestrator instance (provides retrieval
            and LLM chain access).
        question: The raw user query.
        plan: Validated planner output conforming to the v1 contract.

    Returns:
        Response dict with ``answer``, ``answer_type``, ``sources``, and
        an internal ``_planner_meta`` dict for the caller.
    """
    steps = plan["execution_plan"]

    # ------------------------------------------------------------------
    # Temporal intent override: force WEB_SEARCH → LLM, disable RAG
    # ------------------------------------------------------------------
    temporal_query = _detect_temporal_intent(question)
    if temporal_query:
        steps = [
            {"tool": "WEB_SEARCH", "goal": "retrieve"},
            {"tool": "LLM", "goal": "answer"},
        ]
        logger.info(
            "[PLANNER] Temporal intent detected — forcing WEB_SEARCH + LLM, disabling RAG"
        )

    # Accumulated evidence across retrieval steps
    rag_docs: List[str] = []
    rag_metas: List[Dict] = []
    web_docs: List[Dict] = []
    answer = ""

    # Observability: track which tools actually ran
    tools_executed: List[str] = []
    had_retrieval_steps = False

    # Evidence-adequacy override: when the evaluator decides LLM_ONLY,
    # subsequent retrieval steps (WEB_SEARCH) are skipped.
    adequacy_forces_llm_only = False
    adequacy_result: Optional[Dict[str, Any]] = None

    # Resolve company for dynamic collection (Rule 1, 4, 7)
    try:
        company_name = extract_company_name(question)
        if not company_name:
            raise ValueError("Company name could not be resolved.")
    except Exception as e:
        logger.error(f"[PLANNER] Extraction failed: {e}")
        # In planner mode, we might want to return an error response immediately
        return {
            "answer": "Company name could not be resolved. Please specify a company clearly.",
            "answer_type": "ERROR",
            "sources": [],
            "confidence_level": "LOW",
            "_planner_meta": {"error": str(e)}
        }

    for idx, step in enumerate(steps):
        tool = step["tool"]
        goal = step["goal"]

        # --- RAG retrieval ---
        if tool == "RAG" and goal == "retrieve":
            # Skip RAG entirely for temporal queries
            if temporal_query:
                logger.debug("[PLANNER] Temporal query — RAG retrieval skipped")
                continue

            had_retrieval_steps = True
            logger.debug("[PLANNER] Step %d: executing RAG retrieve", idx)
            rag_docs, rag_metas, retrieval_metrics = (
                orchestrator._retrieve_documents_hybrid(question, company_name)
            )
            tools_executed.append("RAG")
            logger.debug(
                "[PLANNER] Step %d: RAG retrieved %d documents", idx, len(rag_docs)
            )

            # --- RAG → WEB_SEARCH auto-escalation (0 documents) ---
            if len(rag_docs) == 0:
                # Only escalate if WEB_SEARCH is not already in remaining steps
                remaining_tools = {s["tool"] for s in steps[idx + 1 :]}
                if "WEB_SEARCH" not in remaining_tools:
                    logger.debug(
                        "[PLANNER] Step %d: ESCALATION — RAG returned 0 documents "
                        "and no WEB_SEARCH planned; auto-triggering WEB_SEARCH",
                        idx,
                    )
                    web_docs = orchestrator._retrieve_web_evidence(question, company_name)
                    tools_executed.append("WEB_SEARCH(auto)")
                    logger.debug(
                        "[PLANNER] Step %d: WEB_SEARCH escalation retrieved %d documents",
                        idx,
                        len(web_docs),
                    )
                else:
                    logger.debug(
                        "[PLANNER] Step %d: RAG returned 0 documents; "
                        "WEB_SEARCH already in plan — skipping auto-escalation",
                        idx,
                    )
            else:
                # --- Evidence adequacy evaluation ---
                from rag.evidence_adequacy import EvidenceAdequacyEvaluator

                adequacy = EvidenceAdequacyEvaluator.evaluate(
                    question=question,
                    rag_docs=rag_docs,
                    rag_metas=rag_metas,
                    best_distance=retrieval_metrics.get("best_distance", 1.5),
                    llm=orchestrator.llm,
                    retrieval_metrics=retrieval_metrics,
                )
                adequacy_result = adequacy
                logger.info(
                    "evidence adequacy evaluated",
                    extra={
                        "event_type": "evidence_adequacy_evaluated",
                        "adequacy_decision": adequacy["decision"],
                        "adequacy_score": adequacy["final_score"],
                        "llm_score": adequacy["llm_score"],
                        "retrieval_score": adequacy["retrieval_score"],
                        "doc_type_score": adequacy["doc_type_score"],
                    },
                )
                logger.info(
                    "confidence_gating_decision",
                    extra={
                        "event_type": "confidence_gating",
                        "decision": adequacy["decision"],
                        "final_score": adequacy["final_score"],
                        "llm_score": adequacy["llm_score"],
                        "retrieval_score": adequacy["retrieval_score"],
                        "doc_type_score": adequacy["doc_type_score"],
                    },
                )

                if adequacy["decision"] == "ESCALATE_WEB":
                    # RAG docs are marginal — supplement with web search
                    remaining_tools = {s["tool"] for s in steps[idx + 1 :]}
                    if "WEB_SEARCH" not in remaining_tools:
                        logger.debug(
                            "[PLANNER] Step %d: ESCALATE_WEB — "
                            "auto-triggering supplementary web search",
                            idx,
                        )
                        web_docs = orchestrator._retrieve_web_evidence(question, company_name)
                        tools_executed.append("WEB_SEARCH(adequacy)")
                        logger.debug(
                            "[PLANNER] Step %d: supplementary WEB_SEARCH "
                            "retrieved %d documents",
                            idx,
                            len(web_docs),
                        )

                elif adequacy["decision"] == "LLM_ONLY":
                    # RAG evidence is too poor to use — discard and
                    # force LLM-only (skip subsequent WEB_SEARCH too)
                    rag_docs = []
                    rag_metas = []
                    adequacy_forces_llm_only = True
                    logger.debug(
                        "[PLANNER] Step %d: LLM_ONLY — "
                        "RAG evidence discarded by adequacy evaluator",
                        idx,
                    )

                # USE_RAG: no action needed — proceed with retrieved docs

        # --- Web search retrieval ---
        elif tool == "WEB_SEARCH" and goal == "retrieve":
            if adequacy_forces_llm_only:
                logger.debug(
                    "[PLANNER] Step %d: WEB_SEARCH skipped "
                    "(adequacy evaluator forced LLM_ONLY)",
                    idx,
                )
                continue

            had_retrieval_steps = True
            logger.debug("[PLANNER] Step %d: executing WEB_SEARCH retrieve", idx)
            web_docs = orchestrator._retrieve_web_evidence(question, company_name)
            tools_executed.append("WEB_SEARCH")
            logger.debug(
                "[PLANNER] Step %d: WEB_SEARCH retrieved %d documents",
                idx,
                len(web_docs),
            )

            # Fallback: if WEB_SEARCH returned nothing, try RAG once
            # DISABLED for temporal queries — RAG must never answer real-time questions
            if not web_docs and not rag_docs and not temporal_query:
                logger.debug(
                    "[PLANNER] WEB_SEARCH returned 0 documents — attempting RAG fallback"
                )
                rag_docs, rag_metas, _ = orchestrator._retrieve_documents_hybrid(question, company_name)
                tools_executed.append("RAG(fallback)")
                logger.debug(
                    "[PLANNER] RAG fallback retrieved %d documents",
                    len(rag_docs),
                )
            elif not web_docs and temporal_query:
                logger.debug(
                    "[PLANNER] Temporal query — RAG fallback skipped"
                )

        # --- LLM answer generation ---
        elif tool == "LLM" and goal == "answer":
            logger.debug("[PLANNER] Step %d: executing LLM answer", idx)

            # -- Answerability gate --
            answerable, gate_reason = _is_answerable(
                question, rag_docs, rag_metas, web_docs,
            )
            if not answerable:
                logger.warning(
                    "[GUARDRAIL] Answerability gate BLOCKED answer: %s",
                    gate_reason,
                )
                MetricsRecorder.record_blocked_query(
                    question, gate_reason,
                    has_documents=bool(rag_docs) or bool(web_docs),
                )
                tools_executed.append("LLM(blocked)")
                return {
                    "answer": (
                        "No verified documents matching the requested "
                        "company, time period, and metrics were found. "
                        "Exact figures cannot be provided."
                    ),
                    "answer_type": "NO_ANSWER",
                    "sources": [],
                    "confidence_level": "LOW",
                    "disclosure_note": gate_reason,
                    "_planner_meta": {
                        "has_evidence": bool(rag_docs) or bool(web_docs),
                        "had_retrieval_steps": had_retrieval_steps,
                        "execution_path": "GUARDRAIL_BLOCKED",
                        "tools_executed": tools_executed,
                    },
                }

            has_evidence = bool(rag_docs) or bool(web_docs)

            if has_evidence:
                # Fuse evidence from all retrieval steps
                fusion_result = EvidenceFusion.fuse_evidence(
                    rag_docs, rag_metas, web_docs, question
                )
                fused_context = fusion_result.get("fused_context", "")

                set_llm_cost_stage("answer_generation")
                if fused_context:
                    answer = orchestrator.rag_chain.invoke(
                        {"question": question, "context": fused_context},
                        config={"callbacks": [cost_tracker_callback]},
                    )
                else:
                    # Fallback: assemble context manually from RAG docs
                    context_parts: List[str] = []
                    for doc, meta in zip(rag_docs, rag_metas):
                        meta_info = ""
                        if meta.get("source"):
                            meta_info = f"Source: {meta['source']}"
                        if meta.get("fiscal_year"):
                            meta_info += f", FY: {meta['fiscal_year']}"
                        if meta.get("page"):
                            meta_info += f", Page: {meta['page']}"
                        context_parts.append(
                            f"[{meta_info}]\n{doc}" if meta_info else doc
                        )
                    answer = orchestrator.rag_chain.invoke(
                        {
                            "question": question,
                            "context": "\n\n---\n\n".join(context_parts),
                        },
                        config={"callbacks": [cost_tracker_callback]},
                    )
                logger.debug(
                    "[PLANNER] Step %d: LLM generated answer from evidence", idx
                )
            else:
                MetricsRecorder.record_llm_fallback(question, reason="no_evidence_planner")
                set_llm_cost_stage("answer_generation")
                answer = orchestrator.llm_only_chain.invoke(
                    {"question": question},
                    config={"callbacks": [cost_tracker_callback]},
                )
                # Temporal queries with no web evidence get a real-time disclaimer
                if temporal_query:
                    _TEMPORAL_DISCLAIMER = (
                        "\n\n---\n⚠️ Real-time data could not be verified "
                        "via live sources at the time of the request."
                    )
                    answer = answer.rstrip() + _TEMPORAL_DISCLAIMER
                    logger.debug(
                        "[PLANNER] Step %d: LLM answer with temporal disclaimer (no web evidence)", idx
                    )
                else:
                    logger.debug(
                        "[PLANNER] Step %d: LLM generated answer (no evidence)", idx
                    )
            tools_executed.append("LLM")

        else:
            logger.warning(
                "[PLANNER] Step %d: skipped unrecognised tool=%s goal=%s",
                idx,
                tool,
                goal,
            )

    # --- Determine execution path (tool-driven, deterministic) ---
    # Strip auto-escalation / fallback suffixes so the check is clean.
    _tool_names = {t.split("(")[0] for t in tools_executed}

    if _tool_names == {"LLM"}:
        execution_path = "LLM_ONLY"
        answer_type = "sagealpha_rag"
    elif "WEB_SEARCH" in _tool_names and "RAG" in _tool_names:
        execution_path = "WEB_SEARCH+RAG"
        answer_type = "sagealpha_hybrid_search"
    elif "WEB_SEARCH" in _tool_names:
        execution_path = "WEB_SEARCH"
        answer_type = "sagealpha_ai_search"
    elif "RAG" in _tool_names:
        execution_path = "RAG"
        answer_type = "sagealpha_rag"
    else:
        execution_path = "UNKNOWN"
        answer_type = "sagealpha_rag"

    logger.debug(
        "[PLANNER] execution_path resolved from tools_executed=%s", tools_executed
    )

    has_evidence = bool(rag_docs) or bool(web_docs)

    # --- Override execution_path when adequacy evaluator forced LLM-only ---
    # RAG was *attempted* (so "RAG" is in tools_executed) but the evidence
    # was discarded.  The actual answer came from llm_only_chain, so the
    # execution_path, confidence, and disclaimer logic must reflect that.
    if adequacy_forces_llm_only and not has_evidence:
        execution_path = "LLM_ONLY"
        answer_type = "sagealpha_rag"
        logger.debug(
            "[PLANNER] execution_path overridden to LLM_ONLY "
            "(adequacy evaluator discarded all evidence)"
        )

    # --- Stronger disclaimer for numeric/financial LLM-only answers ---
    if execution_path == "LLM_ONLY" and _detect_numeric_intent(question):
        _FINANCIAL_DISCLAIMER = (
            "\n\n---\n⚠️ **Note:** This answer is based on general "
            "knowledge and may not reflect audited or finalized "
            "financial figures. For authoritative values, please "
            "consult the company's official investor relations "
            "filings or regulatory disclosures."
        )
        answer = answer.rstrip() + _FINANCIAL_DISCLAIMER
        logger.debug("[PLANNER] Appended financial disclaimer (LLM_ONLY + numeric intent)")

    # --- Normalize sources ---
    formatted_sources = _normalize_sources(rag_metas, web_docs, execution_path)

    # --- Structured observability summary ---
    _extra: Dict[str, Any] = {
        "event_type": "planner_execution_summary",
        "execution_path": execution_path,
        "tools_executed": tools_executed,
        "rag_doc_count": len(rag_docs),
        "web_doc_count": len(web_docs),
        "answer_type": answer_type,
    }
    _rid = get_request_id()
    if _rid is not None:
        _extra["request_id"] = _rid
    logger.info(
        "planner execution complete",
        extra=_extra,
    )

    from rag.telemetry import cost_tracker
    telemetry = cost_tracker.summary()
    logger.info(
        "llm_cost_summary",
        extra={
            "event_type": "llm_cost_summary",
            "total_input_tokens": telemetry["total_input_tokens"],
            "total_output_tokens": telemetry["total_output_tokens"],
            "total_cost_usd": round(telemetry["total_cost_usd"], 6),
            "stages": telemetry["stages"],
        },
    )

    confidence_level, disclosure_note = _derive_confidence_and_disclosure(
        execution_path
    )
    audit_meta = _build_audit_meta(question, execution_path, tools_executed, web_docs)

    # Attach adequacy scores to audit metadata (debug mode only).
    # audit_meta is None when debug is off, so this block is a no-op
    # in production.  No document content is included — scores only.
    if audit_meta is not None and adequacy_result is not None:
        _acfg = getattr(get_config(), "evidence_adequacy", None) or {}
        audit_meta["evidence_adequacy"] = {
            "llm_score": adequacy_result["llm_score"],
            "retrieval_score": adequacy_result["retrieval_score"],
            "doc_type_score": adequacy_result["doc_type_score"],
            "final_score": adequacy_result["final_score"],
            "decision": adequacy_result["decision"],
            "version": _acfg.get("version", "v1_hybrid"),
        }

    result: Dict[str, Any] = {
        "answer": answer,
        "answer_type": answer_type,
        "sources": formatted_sources,
        "confidence_level": confidence_level,
        "disclosure_note": disclosure_note,
        # Internal metadata — popped by answer_query_simple() before
        # returning to the API.  Never reaches the user.
        "_planner_meta": {
            "has_evidence": has_evidence,
            "had_retrieval_steps": had_retrieval_steps,
            "execution_path": execution_path,
            "tools_executed": tools_executed,
        },
    }
    if audit_meta is not None:
        result["_meta"] = audit_meta
    if has_evidence:
        MetricsRecorder.record_rag_answer(question, has_documents=True)
    return result


def answer_query_simple(question: str) -> Dict[str, Any]:
    """Simplified interface for the API layer.

    When ``ENABLE_QUERY_PLANNER`` is ``"true"``, routes the query through
    the LLM-based planner (``plan_query`` → ``_execute_planner_plan``).
    Falls back to the legacy ``answer_query()`` path if the planner fails
    or when the feature flag is disabled.

    Safety / hardening behaviours:
        * Single re-plan: if the first plan's retrieval steps yield zero
          evidence, the planner is re-invoked exactly **once** with the
          hint *"Previous retrieval returned no results"*.  A second
          re-plan is never attempted regardless of outcome.
        * Internal ``_planner_meta`` dict is always stripped before the
          response reaches the caller / API.
        * Any uncaught exception in the planner path falls through to
          legacy routing automatically.
    """
    # ------------------------------------------------------------------ #
    # Hard override: system-introspection queries                         #
    # ------------------------------------------------------------------ #
    if _detect_system_introspection(question):
        logger.info("[SYSTEM] Introspection query handled by system override")
        confidence_level, disclosure_note = _derive_confidence_and_disclosure("SYSTEM")
        audit_meta = _build_audit_meta(question, "SYSTEM", [], [])
        result: Dict[str, Any] = {
            "answer": _SYSTEM_CAPABILITY_RESPONSE,
            "answer_type": "system_capability",
            "sources": _normalize_sources([], [], "SYSTEM"),
            "confidence_level": confidence_level,
            "disclosure_note": disclosure_note,
        }
        if audit_meta is not None:
            result["_meta"] = audit_meta
        return result

    # ------------------------------------------------------------------ #
    # Feature flag: planner-driven routing                                #
    # ------------------------------------------------------------------ #
    planner_enabled = os.getenv("ENABLE_QUERY_PLANNER", "false").lower() == "true"

    if planner_enabled:
        try:
            orchestrator = get_orchestrator()
            plan = plan_query(question)
            result = _execute_planner_plan(orchestrator, question, plan)

            # Single re-plan if first plan yielded no evidence
            planner_meta = result.pop("_planner_meta", {})
            if (
                planner_meta.get("had_retrieval_steps")
                and not planner_meta.get("has_evidence")
            ):
                logger.info(
                    "[PLANNER] First plan returned no evidence — re-planning once"
                )
                plan2 = plan_query(
                    question + " (Previous retrieval returned no results)"
                )
                result = _execute_planner_plan(orchestrator, question, plan2)
                result.pop("_planner_meta", None)

            return result
        except Exception as exc:
            logger.exception(
                "[PLANNER] HARD FAILURE — planner execution aborted (%s: %s)",
                type(exc).__name__,
                exc,
            )
            logger.info("[PLANNER] Falling back to legacy routing after planner failure")
    else:
        logger.info("[PLANNER] Query planner is DISABLED, using legacy routing")

    # ------------------------------------------------------------------ #
    # Legacy path (original behaviour, unchanged)                         #
    # ------------------------------------------------------------------ #
    try:
        orchestrator = get_orchestrator()
        return orchestrator.answer_query(question)
    except RuntimeError as e:
        # Handle initialization errors
        error_msg = str(e)
        logger.error(f"[API] Orchestrator error: {error_msg}")
        
        # Return a user-friendly error response
        if "ChromaDB collection is EMPTY" in error_msg or "FATAL ERROR" in error_msg:
            return {
                "answer": "The knowledge base is not available. Please ensure documents have been ingested by running the ingestion process.",
                "answer_type": "sagealpha_rag",
                "sources": []
            }
        elif "CHROMA_API_KEY" in error_msg or "environment variable" in error_msg:
            return {
                "answer": "Service configuration error. Please check that all required environment variables are set correctly.",
                "answer_type": "sagealpha_rag",
                "sources": []
            }
        else:
            return {
                "answer": f"I apologize, but I encountered an error while initializing the system: {error_msg}. Please contact support.",
                "answer_type": "sagealpha_rag",
                "sources": []
            }
