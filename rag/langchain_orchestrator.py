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
from rag.query_constraints import (
    extract_fiscal_year,
    detect_numeric_intent,
    detect_temporal_intent,
    detect_system_introspection,
)
from rag.retrieval_bundle import RetrievalBundle
from rag.execution_state import ExecutionState
from rag.cross_encoder_reranker import rerank_documents
from rag.answerability_model import (
    AnswerabilityFeatures,
    AnswerabilityModel,
)
from rag.numeric_verifier import verify_numeric_consistency
from rag.source_trust import SourceTrustScorer

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


# extract_fiscal_year — removed: canonical version in rag.query_constraints


# DEPRECATED — replaced by planner-based routing (rag/planner.py).
# This function only recognised three OFSS name variants, making it
# impossible to resolve any other company.  The planner removes the
# need for hard-coded entity extraction at the routing layer.
# Legacy routing has been removed.  This function is dead code.
def _extract_entity_from_query(query: str) -> Optional[str]:
    """Extracts company/entity from query.

    .. deprecated:: Replaced by planner-based routing.  Dead code —
       legacy routing pipeline removed.
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
            detect_numeric_intent(question)
            or extract_fiscal_year(question)
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
    requested_year = extract_fiscal_year(question)
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
        # Entity mismatch is NOT a block condition.  When documents carry
        # company metadata but none match, we log a warning and continue
        # with reduced confidence — the caller checks for the
        # "entity_mismatch_warning" prefix in gate_reason.
        if has_company_metadata and not entity_found:
            logger.warning(
                "[GUARDRAIL] Entity mismatch (non-blocking): requested='%s', "
                "available companies=%s — continuing with reduced confidence",
                requested_entity,
                [m.get("company", "") for m in all_metas if m.get("company")],
            )
            return (
                True,
                f"entity_mismatch_warning: no documents match requested "
                f"company '{requested_entity}'",
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
# Legacy routing has been removed.  This function is dead code.
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
    
    requested_year = extract_fiscal_year(query)
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


# detect_numeric_intent — removed: canonical version in rag.query_constraints


# detect_temporal_intent — removed: canonical version in rag.query_constraints


# detect_system_introspection — removed: canonical version in rag.query_constraints


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
        "temporal_override": detect_temporal_intent(question),
        "introspection_override": detect_system_introspection(question),
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
        if not norm_name:
            logger.warning("[BM25] Cannot build index — company name normalizes to None")
            return None

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
    _TOP_K_PER_COLLECTION: int = 10
    _FINAL_TOP_K: int = 10
    _RERANK_POOL_SIZE: int = 20

    def _retrieve_documents_hybrid(self, question: str, company_name: str) -> RetrievalBundle:
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
            "best_cross_score": 0.0,
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
                return RetrievalBundle([], [], _empty_metrics)

            # ---------------------------------------------------------- #
            # 1. Prepare query                                            #
            # ---------------------------------------------------------- #
            requested_year = extract_fiscal_year(question)
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
            # 5. Cross-encoder re-ranking                                 #
            # ---------------------------------------------------------- #
            # Take a broader pool (top _RERANK_POOL_SIZE) for re-ranking,
            # then keep _FINAL_TOP_K after cross-encoder scoring.
            rerank_pool = deduped[: self._RERANK_POOL_SIZE]

            pool_docs = [r[0] for r in rerank_pool]
            pool_metas = [r[1] for r in rerank_pool]
            pool_distances = [r[2] for r in rerank_pool]

            # Preserve cosine distance in metadata for telemetry
            for meta, dist in zip(pool_metas, pool_distances):
                meta["cosine_distance"] = dist

            if pool_docs:
                logger.info(
                    "[RETRIEVER] Pre-rerank pool: %d candidates (cosine distances: %s)",
                    len(pool_docs),
                    [round(d, 4) for d in pool_distances],
                )

            # Cross-encoder re-ranking: produces (doc, meta, cross_score) sorted desc
            reranked = rerank_documents(question, pool_docs, pool_metas)

            # Select final top-k from re-ranked results
            top_reranked = reranked[: self._FINAL_TOP_K]

            chroma_docs = [r[0] for r in top_reranked]
            chroma_metas = [r[1] for r in top_reranked]
            chroma_distances = [
                r[1].get("cosine_distance", 2.0) for r in top_reranked
            ]

            # Extract the collection name from metadata for compatibility
            top_results = [
                (doc, meta, meta.get("cosine_distance", 2.0), meta.get("_source_collection", ""))
                for doc, meta in zip(chroma_docs, chroma_metas)
            ]

            if chroma_distances:
                logger.info(f"[RETRIEVER] Top cosine distances (for telemetry): {chroma_distances}")
                logger.info(f"[RETRIEVER] Best cosine distance: {chroma_distances[0]:.4f}")

            if chroma_metas:
                cross_scores = [m.get("cross_score", 0.0) for m in chroma_metas]
                best_cross = max(cross_scores) if cross_scores else 0.0
                sample = chroma_metas[0]
                logger.info(
                    "[RETRIEVER] Best match (post-rerank) → collection=%s company=%s year=%s cross_score=%.6f",
                    sample.get("_source_collection", "N/A"),
                    sample.get("company", "N/A"),
                    sample.get("fiscal_year", "N/A"),
                    sample.get("cross_score", 0.0),
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
            if detect_numeric_intent(question) and collections_with_hits:
                # Run BM25 on the collection that contributed the best hit
                best_col_name = top_results[0][3] if top_results else None
                if best_col_name:
                    best_collection = collections_by_name.get(best_col_name) or get_company_collection(best_col_name)
                    if best_collection is None:
                        logger.info(
                            "[POLICY_ROUTER] No internal coverage for '%s' — BM25 skipped",
                            best_col_name,
                        )
                    else:
                        pass  # fall through to BM25 try block below
                if best_col_name and best_collection is not None:
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
                if company_norm:
                    company_matches = sum(
                        1 for m in merged_metas
                        if (
                            (meta_norm := normalize_company_name(
                                str(m.get("company") or m.get("_source_collection") or "")
                            ))
                            and meta_norm == company_norm
                        )
                    )
                    company_match_ratio = company_matches / n_metas
                else:
                    company_match_ratio = 1.0
            else:
                company_match_ratio = 1.0

            avg_dist = (
                sum(chroma_distances) / len(chroma_distances)
                if chroma_distances
                else 2.0
            )
            best_sim = max(0.0, 1.0 - (best_dist / 2.0))
            avg_sim = max(0.0, 1.0 - (avg_dist / 2.0))

            # Compute best_cross_score from re-ranked metadata
            _cross_scores = [
                m.get("cross_score", 0.0) for m in merged_metas
                if "cross_score" in m
            ]
            best_cross_score = max(_cross_scores) if _cross_scores else 0.0

            # --- STABILIZATION: Retrieval Quality Threshold (Rule 3) ---
            # When a cross-encoder is active, a strong cross-encoder score
            # overrides a high cosine distance (which can happen when the
            # bi-encoder and cross-encoder disagree on relevance).
            rag_status = "FOUND" if merged_docs else "EMPTY"
            if rag_status == "FOUND":
                if best_cross_score > 0.0:
                    # Cross-encoder active: LOW_CONFIDENCE only if score is weak
                    if best_cross_score < 0.3:
                        logger.warning(
                            "[RETRIEVER] LOW_CONFIDENCE match detected "
                            "(best_cross_score=%.4f < 0.3)",
                            best_cross_score,
                        )
                        rag_status = "LOW_CONFIDENCE"
                else:
                    # Legacy distance-based check
                    if best_dist >= 1.5:
                        logger.warning(
                            "[RETRIEVER] LOW_CONFIDENCE match detected "
                            "(dist=%.4f >= 1.5)",
                            best_dist,
                        )
                        rag_status = "LOW_CONFIDENCE"

            retrieval_metrics: Dict = {
                "best_distance": best_dist,
                "best_cross_score": best_cross_score,
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

            return RetrievalBundle(merged_docs, merged_metas, retrieval_metrics)

        except Exception as e:
            logger.error(f"[RETRIEVER] Multi-collection retrieval failed: {e}", exc_info=True)
            return RetrievalBundle([], [], _empty_metrics)
    
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

    # ------------------------------------------------------------------ #
    # Internal coverage signal (policy-based routing)                      #
    # ------------------------------------------------------------------ #

    def _internal_coverage(self, company_name: str) -> bool:
        """Return ``True`` if the company has a dedicated internal collection.

        Performs a pure collection-name presence check against the vector-
        store — no keyword heuristics, no LLM calls, no vocabulary lists.

        Parameters
        ----------
        company_name : str
            Raw company name as extracted from the user query.

        Returns
        -------
        bool
            ``True`` when a matching collection exists, ``False`` otherwise
            (including when *company_name* is empty or un-normalisable).
        """
        from rag.company_normalizer import normalize_company_name

        collections = list_all_collections(skip_internal=True)

        normalized = normalize_company_name(company_name) if company_name else None
        if not normalized:
            return False

        for c in collections:
            name = getattr(c, "name", str(c))
            if normalized in name:
                return True

        return False

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
        adequacy_forces_llm_only = False

        try:
            logger.info("=" * 60)
            logger.info("[QUERY] Processing query (availability-based routing)")
            logger.info("=" * 60)
            logger.info("[QUERY] Query text: %s", question[:120])

            # -------------------------------------------------------------- #
            # STEP 1: ALWAYS attempt RAG retrieval                            #
            # -------------------------------------------------------------- #
            # Resolve company for metadata-alignment boosting.
            # Company resolution is NEVER fatal — retrieval must always
            # proceed.  An unresolved company only means metadata-alignment
            # boosts are skipped.
            try:
                company_name = extract_company_name(question)
            except Exception as e:
                logger.warning("[ROUTER] Company extraction raised (%s) — ignored", e)
                company_name = ""

            if not company_name:
                logger.warning(
                    "[RETRIEVER] Company resolution failed — proceeding without strict alignment"
                )
                company_name = ""

            logger.info("[STEP 1] Attempting RAG retrieval for ALL queries")
            _rag_bundle = self._retrieve_documents_hybrid(question, company_name)
            documents = list(_rag_bundle.documents)
            metadatas = list(_rag_bundle.metadatas)
            retrieval_metrics = _rag_bundle.metrics
            logger.info("[STEP 1] RAG retrieved %d documents", len(documents))

            # -------------------------------------------------------------- #
            # STEP 2: Answerability-based routing (learned classifier)        #
            # -------------------------------------------------------------- #
            web_documents: List[Dict] = []

            if len(documents) == 0:
                logger.info("[STEP 2] RAG returned 0 documents → triggering WEB_SEARCH")
                web_documents = self._retrieve_web_evidence(question, company_name)
                logger.info("[STEP 2] WEB_SEARCH retrieved %d documents", len(web_documents))
            else:
                _ans_features = AnswerabilityFeatures(
                    best_cross_score=retrieval_metrics.get("best_cross_score", 0.0),
                    doc_count=len(documents),
                    year_match_ratio=retrieval_metrics.get("year_match_ratio", 1.0),
                    entity_match_ratio=retrieval_metrics.get("company_match_ratio", 1.0),
                    numeric_intent=detect_numeric_intent(question),
                )
                adequacy = AnswerabilityModel.predict(_ans_features)
                logger.info(
                    "confidence_gating_decision",
                    extra={
                        "event_type": "confidence_gating",
                        "decision": adequacy["decision"],
                        "probabilities": adequacy["probabilities"],
                        "model_version": adequacy["model_version"],
                    },
                )
                if adequacy["decision"] == "ESCALATE_WEB":
                    logger.info("[STEP 2] Answerability ESCALATE_WEB → triggering WEB_SEARCH")
                    web_documents = self._retrieve_web_evidence(question, company_name)
                    logger.info("[STEP 2] WEB_SEARCH retrieved %d documents", len(web_documents))
                elif adequacy["decision"] == "LLM_ONLY":
                    # RAG evidence is too poor to use for answer generation.
                    # Flag LLM-only routing; do NOT destroy documents/metadatas
                    # so downstream stages still see what was retrieved.
                    adequacy_forces_llm_only = True
                    logger.info(
                        "[STEP 2] Answerability LLM_ONLY — flagged; "
                        "RAG docs preserved (%d docs)", len(documents),
                    )

            # -------------------------------------------------------------- #
            # STEP 3: Determine execution path                                #
            # -------------------------------------------------------------- #
            has_rag = len(documents) > 0 and not adequacy_forces_llm_only
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
                "[ROUTER] rag_docs=%d (usable=%s)  web_docs=%d",
                len(documents),
                not adequacy_forces_llm_only,
                len(web_documents),
            )
            logger.info("=" * 60)

            # -------------------------------------------------------------- #
            # PRE-GUARDRAIL WEB ESCALATION (numeric intent)                   #
            # -------------------------------------------------------------- #
            # When adequacy forced LLM_ONLY but the query has numeric
            # intent, escalate to web BEFORE the answerability gate so
            # it is not blocked for lack of evidence.
            if adequacy_forces_llm_only and not web_documents:
                _has_num = detect_numeric_intent(question)
                _has_fy = bool(extract_fiscal_year(question))
                if _has_num or _has_fy:
                    logger.info(
                        "[ESCALATION] Numeric intent detected — "
                        "escalating to web before guardrail"
                    )
                    web_documents = self._retrieve_web_evidence(
                        question, company_name
                    )
                    if web_documents:
                        has_web = True
                        # Recompute execution path with new web evidence.
                        if has_rag and has_web:
                            execution_path = "RAG+WEB_SEARCH+LLM"
                        elif has_web:
                            execution_path = "WEB_SEARCH+LLM"
                        logger.info(
                            "[ESCALATION] Web escalation retrieved %d "
                            "documents — execution_path=%s",
                            len(web_documents), execution_path,
                        )
                    else:
                        logger.info(
                            "[ESCALATION] Web escalation returned 0 "
                            "documents — proceeding to LLM"
                        )

            # -------------------------------------------------------------- #
            # ANSWERABILITY GATE                                               #
            # -------------------------------------------------------------- #
            # When answerability model forced LLM_ONLY, rag docs are
            # preserved but excluded from the answerability gate (already
            # deemed too poor by the classifier).
            _gate_rag = documents if has_rag else []
            _gate_metas = metadatas if has_rag else []
            answerable, gate_reason = _is_answerable(
                question, _gate_rag, _gate_metas, web_documents,
            )

            # Entity mismatch is non-blocking — reduce confidence but
            # continue to answer generation.
            _entity_mismatch = gate_reason.startswith("entity_mismatch_warning")

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
            # STEP 3b: Multi-source numeric verification                      #
            # -------------------------------------------------------------- #
            _numeric_single_source = False
            if detect_numeric_intent(question) and has_rag:
                _nv_result = verify_numeric_consistency(
                    question, documents, metadatas,
                )
                if _nv_result["status"] == "CONFLICT":
                    logger.info(
                        "[NUMERIC_VERIFIER] CONFLICT detected — "
                        "forcing web escalation"
                    )
                    if not has_web:
                        web_documents = self._retrieve_web_evidence(
                            question, company_name,
                        )
                        if web_documents:
                            has_web = True
                            if has_rag and has_web:
                                execution_path = "RAG+WEB_SEARCH+LLM"
                            elif has_web:
                                execution_path = "WEB_SEARCH+LLM"
                            logger.info(
                                "[NUMERIC_VERIFIER] Web escalation retrieved "
                                "%d documents — execution_path=%s",
                                len(web_documents), execution_path,
                            )
                elif _nv_result["status"] == "SINGLE_SOURCE":
                    _numeric_single_source = True
                    logger.info(
                        "[NUMERIC_VERIFIER] SINGLE_SOURCE — "
                        "confidence will be reduced to LOW"
                    )
                # CONSISTENT → normal flow, no action needed

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
                _llm_reason = (
                    "adequacy_llm_only" if adequacy_forces_llm_only
                    else "no_evidence"
                )
                logger.info("[STEP 4] No usable evidence — using LLM-only with disclaimer")
                MetricsRecorder.record_llm_fallback(question, reason=_llm_reason)
                set_llm_cost_stage("answer_generation")
                answer = self.llm_only_chain.invoke(
                    {"question": question},
                    config={"callbacks": [cost_tracker_callback]},
                )

                # Append disclaimer — stronger for numeric/financial queries
                if detect_numeric_intent(question):
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
            # Always pass original metadatas — retrieval evidence is immutable.
            # _normalize_sources uses execution_path to decide inclusion.
            formatted_sources = _normalize_sources(
                metadatas, web_documents, execution_path
            )

            # -------------------------------------------------------------- #
            # STEP 7: Summary log + return                                    #
            # -------------------------------------------------------------- #
            logger.info("=" * 60)
            logger.info("[RESPONSE] ROUTING SUMMARY")
            logger.info("=" * 60)
            logger.info("[RESPONSE]   execution_path = %s", execution_path)
            logger.info("[RESPONSE]   answer_type    = %s", answer_type)
            logger.info("[RESPONSE]   rag_docs       = %d (usable=%s)", len(documents), not adequacy_forces_llm_only)
            logger.info("[RESPONSE]   web_docs       = %d", len(web_documents))
            logger.info(
                "[RESPONSE]   sources_count  = %d", len(formatted_sources),
            )
            logger.info("=" * 60)

            # -------------------------------------------------------------- #
            # STEP 8: Post-Processing & Validation                           #
            # -------------------------------------------------------------- #
            # 1. Confidence Scoring — always receives original metadatas
            #    (retrieval evidence is immutable after retrieval stage).
            confidence_score = ConfidenceScorer.calculate_score(
                execution_path=execution_path,
                rag_metas=metadatas,
                web_docs=web_documents
            )
            confidence_level = ConfidenceScorer.get_confidence_level(confidence_score)

            # Entity mismatch reduces confidence to LOW regardless of score.
            if _entity_mismatch:
                confidence_level = "LOW"
                logger.info(
                    "[GUARDRAIL] Confidence reduced to LOW due to entity mismatch"
                )

            # Single-source numeric verification reduces confidence to LOW.
            if _numeric_single_source and confidence_level != "LOW":
                confidence_level = "LOW"
                logger.info(
                    "[NUMERIC_VERIFIER] Confidence reduced to LOW — "
                    "numeric value from single source only"
                )

            # Get disclosure note from legacy logic
            _, disclosure_note = _derive_confidence_and_disclosure(execution_path)

            if _entity_mismatch:
                disclosure_note = (
                    "Answer may not match the exact requested company. "
                    "Verify against official filings."
                )

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
    temporal_query = detect_temporal_intent(question)
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

    # Answerability override: when the classifier decides LLM_ONLY,
    # subsequent retrieval steps (WEB_SEARCH) are skipped.
    adequacy_forces_llm_only = False
    adequacy_result: Optional[Dict[str, Any]] = None

    # Numeric verification: set when a numeric query has only a single
    # source backing the value — triggers confidence reduction to LOW.
    _numeric_single_source = False

    # Entity mismatch flag — set by answerability gate when entity
    # constraint fails.  Reduces confidence but does not block answer.
    _entity_mismatch = False

    # Resolve company for metadata-alignment boosting.
    # Company resolution is NEVER fatal — retrieval must always proceed.
    # An unresolved company only means metadata-alignment boosts are skipped.
    try:
        company_name = extract_company_name(question)
    except Exception as e:
        logger.warning("[PLANNER] Company extraction raised (%s) — ignored", e)
        company_name = ""

    if not company_name:
        logger.warning(
            "[RETRIEVER] Company resolution failed — proceeding without strict alignment"
        )
        company_name = ""

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
            _rag_bundle = orchestrator._retrieve_documents_hybrid(question, company_name)
            rag_docs = list(_rag_bundle.documents)
            rag_metas = list(_rag_bundle.metadatas)
            retrieval_metrics = _rag_bundle.metrics
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
                # --- Answerability model (learned classifier) ---
                _ans_features = AnswerabilityFeatures(
                    best_cross_score=retrieval_metrics.get("best_cross_score", 0.0),
                    doc_count=len(rag_docs),
                    year_match_ratio=retrieval_metrics.get("year_match_ratio", 1.0),
                    entity_match_ratio=retrieval_metrics.get("company_match_ratio", 1.0),
                    numeric_intent=detect_numeric_intent(question),
                )
                adequacy = AnswerabilityModel.predict(_ans_features)
                adequacy_result = adequacy
                logger.info(
                    "answerability_evaluated",
                    extra={
                        "event_type": "answerability_evaluated",
                        "decision": adequacy["decision"],
                        "probabilities": adequacy["probabilities"],
                        "model_version": adequacy["model_version"],
                    },
                )
                logger.info(
                    "confidence_gating_decision",
                    extra={
                        "event_type": "confidence_gating",
                        "decision": adequacy["decision"],
                        "probabilities": adequacy["probabilities"],
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
                        tools_executed.append("WEB_SEARCH(answerability)")
                        logger.debug(
                            "[PLANNER] Step %d: supplementary WEB_SEARCH "
                            "retrieved %d documents",
                            idx,
                            len(web_docs),
                        )

                elif adequacy["decision"] == "LLM_ONLY":
                    # RAG evidence is too poor to use for answer generation.
                    # Flag LLM-only routing; do NOT destroy rag_docs/rag_metas
                    # so downstream stages (source normalization, confidence
                    # scoring, audit metadata) still see what was retrieved.
                    adequacy_forces_llm_only = True
                    logger.debug(
                        "[PLANNER] Step %d: LLM_ONLY — "
                        "answerability model flagged; rag_docs preserved (%d docs)",
                        idx,
                        len(rag_docs),
                    )

                # USE_RAG: no action needed — proceed with retrieved docs

        # --- Web search retrieval ---
        elif tool == "WEB_SEARCH" and goal == "retrieve":
            if adequacy_forces_llm_only:
                logger.debug(
                    "[PLANNER] Step %d: WEB_SEARCH skipped "
                    "(answerability model forced LLM_ONLY)",
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
                _fallback_bundle = orchestrator._retrieve_documents_hybrid(question, company_name)
                rag_docs = list(_fallback_bundle.documents)
                rag_metas = list(_fallback_bundle.metadatas)
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

            # ----------------------------------------------------------
            # PRE-GUARDRAIL WEB ESCALATION for numeric intent
            # ----------------------------------------------------------
            # When adequacy forced LLM_ONLY but the query has numeric
            # intent, escalate to web BEFORE the answerability gate.
            # This prevents the guardrail from blocking a numeric query
            # that could be answered with web evidence.
            if adequacy_forces_llm_only and not web_docs:
                _has_num = detect_numeric_intent(question)
                _has_fy = bool(extract_fiscal_year(question))
                if _has_num or _has_fy:
                    logger.info(
                        "[ESCALATION] Numeric intent detected — "
                        "escalating to web before guardrail"
                    )
                    web_docs = orchestrator._retrieve_web_evidence(
                        question, company_name
                    )
                    if web_docs:
                        tools_executed.append("WEB_SEARCH(numeric_escalation)")
                        logger.info(
                            "[ESCALATION] Web escalation retrieved %d "
                            "documents",
                            len(web_docs),
                        )
                    else:
                        logger.info(
                            "[ESCALATION] Web escalation returned 0 "
                            "documents — proceeding to LLM"
                        )

            # -- Answerability gate --
            # When adequacy forced LLM_ONLY, rag_docs are preserved but
            # must not influence the answerability decision (the adequacy
            # evaluator already decided the evidence is too poor).
            _gate_rag = rag_docs if not adequacy_forces_llm_only else []
            _gate_metas = rag_metas if not adequacy_forces_llm_only else []
            answerable, gate_reason = _is_answerable(
                question, _gate_rag, _gate_metas, web_docs,
            )

            # Entity mismatch is non-blocking — flag for confidence
            # reduction but continue to answer generation.
            _entity_mismatch = gate_reason.startswith("entity_mismatch_warning")

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

            # -- Multi-source numeric verification --
            _numeric_single_source = False
            _has_usable_rag = bool(rag_docs) and not adequacy_forces_llm_only
            if detect_numeric_intent(question) and _has_usable_rag:
                _nv_result = verify_numeric_consistency(
                    question, rag_docs, rag_metas,
                )
                if _nv_result["status"] == "CONFLICT":
                    logger.info(
                        "[NUMERIC_VERIFIER] CONFLICT detected — "
                        "forcing web escalation"
                    )
                    if not web_docs:
                        web_docs = orchestrator._retrieve_web_evidence(
                            question, company_name,
                        )
                        if web_docs:
                            tools_executed.append("WEB_SEARCH(numeric_conflict)")
                            logger.info(
                                "[NUMERIC_VERIFIER] Web escalation retrieved "
                                "%d documents",
                                len(web_docs),
                            )
                elif _nv_result["status"] == "SINGLE_SOURCE":
                    _numeric_single_source = True
                    logger.info(
                        "[NUMERIC_VERIFIER] SINGLE_SOURCE — "
                        "confidence will be reduced to LOW"
                    )
                # CONSISTENT → normal flow, no action needed

            # Determine whether usable evidence exists for answer generation.
            # adequacy_forces_llm_only means RAG docs were retrieved but are
            # too poor for fusion — they are kept for audit/confidence only.
            usable_evidence = (
                (bool(rag_docs) and not adequacy_forces_llm_only)
                or bool(web_docs)
            )

            if usable_evidence:
                # Fuse evidence from all retrieval steps.
                # When adequacy forced LLM_ONLY, rag_docs are excluded
                # from fusion (they are preserved only for downstream
                # source normalization, confidence scoring, and audit).
                fusion_rag = rag_docs if not adequacy_forces_llm_only else []
                fusion_metas = rag_metas if not adequacy_forces_llm_only else []
                fusion_result = EvidenceFusion.fuse_evidence(
                    fusion_rag, fusion_metas, web_docs, question
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
                    for doc, meta in zip(fusion_rag, fusion_metas):
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
                # LLM-only path: no usable evidence (either nothing
                # retrieved, or adequacy flagged all RAG docs as too poor).
                _llm_reason = (
                    "adequacy_llm_only" if adequacy_forces_llm_only
                    else "no_evidence_planner"
                )
                MetricsRecorder.record_llm_fallback(question, reason=_llm_reason)
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

    # ------------------------------------------------------------------ #
    # Resolve execution state (deterministic routing contract)             #
    # ------------------------------------------------------------------ #
    # Usable RAG count is 0 when adequacy forced LLM_ONLY — raw rag_docs
    # are preserved for audit/confidence but must not influence routing.
    _usable_rag_count = len(rag_docs) if not adequacy_forces_llm_only else 0

    execution_state = ExecutionState(
        retrieval_count=_usable_rag_count,
        web_count=len(web_docs),
        adequacy_decision=(
            adequacy_result["decision"] if adequacy_result else "N/A"
        ),
        numeric_intent=detect_numeric_intent(question),
        requested_year=extract_fiscal_year(question),
    )
    execution_state.resolve_mode()

    # Derive legacy-compatible strings from the governance state.
    execution_path = execution_state.execution_path
    answer_type = execution_state.answer_type
    has_evidence = bool(rag_docs) or bool(web_docs)

    logger.info("execution_contract", extra=execution_state.to_log_dict())
    logger.debug(
        "[PLANNER] execution_state resolved: %s  (legacy execution_path=%s)",
        execution_state,
        execution_path,
    )

    # --- Stronger disclaimer for numeric/financial LLM-only answers ---
    if execution_state.final_mode == "LLM_ONLY" and execution_state.numeric_intent:
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
        "final_mode": execution_state.final_mode,
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

    # Entity mismatch reduces confidence to LOW regardless of path.
    if _entity_mismatch:
        confidence_level = "LOW"
        disclosure_note = (
            "Answer may not match the exact requested company. "
            "Verify against official filings."
        )
        logger.info(
            "[GUARDRAIL] Confidence reduced to LOW due to entity mismatch"
        )

    # Single-source numeric verification reduces confidence to LOW.
    if _numeric_single_source and confidence_level != "LOW":
        confidence_level = "LOW"
        logger.info(
            "[NUMERIC_VERIFIER] Confidence reduced to LOW — "
            "numeric value from single source only"
        )

    audit_meta = _build_audit_meta(question, execution_path, tools_executed, web_docs)

    # Attach answerability model output to audit metadata (debug mode
    # only).  audit_meta is None when debug is off, so this block is
    # a no-op in production.  No document content is included.
    if audit_meta is not None and adequacy_result is not None:
        audit_meta["answerability"] = {
            "decision": adequacy_result["decision"],
            "probabilities": adequacy_result.get("probabilities", {}),
            "logits": adequacy_result.get("logits", {}),
            "features": adequacy_result.get("features", {}),
            "model_version": adequacy_result.get("model_version", "unknown"),
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
            "final_mode": execution_state.final_mode,
            "tools_executed": tools_executed,
        },
    }
    if audit_meta is not None:
        result["_meta"] = audit_meta
    if has_evidence:
        MetricsRecorder.record_rag_answer(question, has_documents=True)
    return result


def answer_query_simple(question: str) -> Dict[str, Any]:
    """Policy-based entry point for the API layer.

    Routes queries **deterministically** based on internal collection
    coverage — no LLM-based planner, no ``plan_query`` invocation.

    Execution policies
    ------------------
    ``RAG_FIRST``
        The company has a dedicated internal collection.  Retrieve from
        the vectorstore, evaluate answerability, escalate to web search
        only when the answerability model signals ``ESCALATE_WEB``.

    ``WEB_FIRST``
        No internal collection (or no company detected, or temporal
        intent).  Retrieve from the web; fall back to RAG if the web
        search returns nothing (unless the query is temporal).

    Safety / hardening behaviours
    -----------------------------
    * **NO_ANSWER safety net** — if the pipeline produces a
      ``NO_ANSWER`` (e.g. guardrail block), an LLM-only fallback is
      invoked to guarantee output.
    * All error handling (``RuntimeError``, generic ``Exception``) is
      preserved.
    * ``_execute_planner_plan`` and ``plan_query`` are **not** called.
    """
    # ------------------------------------------------------------------ #
    # Hard override: system-introspection queries                         #
    # ------------------------------------------------------------------ #
    if detect_system_introspection(question):
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
    # Policy-based routing (deterministic — no LLM planner)               #
    # ------------------------------------------------------------------ #
    try:
        orchestrator = get_orchestrator()

        # --- Temporal intent ---
        temporal_query = detect_temporal_intent(question)

        # --- Company extraction (never fatal) ---
        try:
            company_name = extract_company_name(question)
        except Exception as e:
            logger.warning(
                "[POLICY_ROUTER] Company extraction raised (%s) — ignored", e
            )
            company_name = ""
        if not company_name:
            company_name = ""

        # --- Internal coverage signal ---
        has_internal = orchestrator._internal_coverage(company_name)

        if temporal_query:
            execution_policy = "WEB_FIRST"
        elif has_internal:
            execution_policy = "RAG_FIRST"
        else:
            execution_policy = "WEB_FIRST"

        logger.info(
            "[POLICY_ROUTER] execution_policy=%s company=%s "
            "coverage=%s temporal=%s",
            execution_policy,
            company_name,
            has_internal,
            temporal_query,
        )
        logger.info("[ROUTING_CONTRACT] resilient=True")

        # --- Accumulated evidence across retrieval steps ---
        rag_docs: List[str] = []
        rag_metas: List[Dict] = []
        web_docs: List[Dict] = []
        answer = ""
        tools_executed: List[str] = []
        adequacy_forces_llm_only = False
        adequacy_result: Optional[Dict[str, Any]] = None
        _numeric_single_source = False
        _entity_mismatch = False
        retrieval_metrics: Dict[str, Any] = {}

        # ============================================================== #
        #  RAG_FIRST path                                                 #
        # ============================================================== #
        if execution_policy == "RAG_FIRST":
            _rag_bundle = orchestrator._retrieve_documents_hybrid(
                question, company_name
            )
            rag_docs = list(_rag_bundle.documents)
            rag_metas = list(_rag_bundle.metadatas)
            retrieval_metrics = _rag_bundle.metrics
            tools_executed.append("RAG")

            logger.info(
                "[POLICY_ROUTER] RAG retrieved %d documents", len(rag_docs)
            )

            if len(rag_docs) == 0:
                # RAG empty — escalate to web
                web_docs = orchestrator._retrieve_web_evidence(
                    question, company_name
                )
                tools_executed.append("WEB_SEARCH(rag_empty)")
                logger.info(
                    "[POLICY_ROUTER] RAG empty — web escalation "
                    "retrieved %d documents",
                    len(web_docs),
                )
            else:
                # --- Answerability model (learned classifier) ---
                _ans_features = AnswerabilityFeatures(
                    best_cross_score=retrieval_metrics.get(
                        "best_cross_score", 0.0
                    ),
                    doc_count=len(rag_docs),
                    year_match_ratio=retrieval_metrics.get(
                        "year_match_ratio", 1.0
                    ),
                    entity_match_ratio=retrieval_metrics.get(
                        "company_match_ratio", 1.0
                    ),
                    numeric_intent=detect_numeric_intent(question),
                )
                adequacy = AnswerabilityModel.predict(_ans_features)
                adequacy_result = adequacy

                logger.info(
                    "answerability_evaluated",
                    extra={
                        "event_type": "answerability_evaluated",
                        "decision": adequacy["decision"],
                        "probabilities": adequacy["probabilities"],
                        "model_version": adequacy["model_version"],
                    },
                )
                logger.info(
                    "[POLICY_ROUTER] routing_decision=%s",
                    adequacy["decision"],
                )

                if adequacy["decision"] == "ESCALATE_WEB":
                    web_docs = orchestrator._retrieve_web_evidence(
                        question, company_name
                    )
                    tools_executed.append("WEB_SEARCH(answerability)")
                    logger.info(
                        "[POLICY_ROUTER] ESCALATE_WEB — web search "
                        "retrieved %d documents",
                        len(web_docs),
                    )
                elif adequacy["decision"] == "LLM_ONLY":
                    adequacy_forces_llm_only = True
                    logger.info(
                        "[POLICY_ROUTER] LLM_ONLY — RAG docs preserved "
                        "(%d docs) but excluded from fusion",
                        len(rag_docs),
                    )
                # USE_RAG: no action needed — proceed with retrieved docs

        # ============================================================== #
        #  WEB_FIRST path                                                 #
        # ============================================================== #
        elif execution_policy == "WEB_FIRST":
            web_docs = orchestrator._retrieve_web_evidence(
                question, company_name
            )
            tools_executed.append("WEB_SEARCH")

            logger.info(
                "[POLICY_ROUTER] WEB_SEARCH retrieved %d documents",
                len(web_docs),
            )

            # Fallback to RAG if web empty (skip for temporal queries)
            if not web_docs and not temporal_query:
                _rag_bundle = orchestrator._retrieve_documents_hybrid(
                    question, company_name
                )
                rag_docs = list(_rag_bundle.documents)
                rag_metas = list(_rag_bundle.metadatas)
                retrieval_metrics = _rag_bundle.metrics
                tools_executed.append("RAG(fallback)")
                logger.info(
                    "[POLICY_ROUTER] Web empty — RAG fallback "
                    "retrieved %d documents",
                    len(rag_docs),
                )
            elif not web_docs and temporal_query:
                logger.info(
                    "[POLICY_ROUTER] Temporal query — RAG fallback skipped"
                )

        # ============================================================== #
        #  Attach trust scores to all retrieved documents                 #
        # ============================================================== #
        for _m in rag_metas:
            _m["trust_score"] = SourceTrustScorer.score(_m)
        for _wd in web_docs:
            _wd_meta = _wd.get("metadata", _wd)
            _wd_meta["trust_score"] = SourceTrustScorer.score(_wd_meta)

        # ============================================================== #
        #  Pre-guardrail web escalation for numeric intent                #
        # ============================================================== #
        if adequacy_forces_llm_only and not web_docs:
            _has_num = detect_numeric_intent(question)
            _has_fy = bool(extract_fiscal_year(question))
            if _has_num or _has_fy:
                logger.info(
                    "[ESCALATION] Numeric intent detected — "
                    "escalating to web before guardrail"
                )
                web_docs = orchestrator._retrieve_web_evidence(
                    question, company_name
                )
                if web_docs:
                    # Attach trust scores to late-arriving web docs
                    for _wd in web_docs:
                        _wd_meta = _wd.get("metadata", _wd)
                        _wd_meta["trust_score"] = SourceTrustScorer.score(
                            _wd_meta
                        )
                    tools_executed.append("WEB_SEARCH(numeric_escalation)")
                    logger.info(
                        "[ESCALATION] Web escalation retrieved %d documents",
                        len(web_docs),
                    )
                else:
                    logger.info(
                        "[ESCALATION] Web escalation returned 0 documents "
                        "— proceeding to LLM"
                    )

        # ============================================================== #
        #  Guardrail: answerability gate                                  #
        # ============================================================== #
        _gate_rag = rag_docs if not adequacy_forces_llm_only else []
        _gate_metas = rag_metas if not adequacy_forces_llm_only else []
        answerable, gate_reason = _is_answerable(
            question, _gate_rag, _gate_metas, web_docs,
        )

        _entity_mismatch = gate_reason.startswith("entity_mismatch_warning")

        if not answerable:
            logger.warning(
                "[GUARDRAIL] Answerability gate BLOCKED answer: %s",
                gate_reason,
            )
            MetricsRecorder.record_blocked_query(
                question,
                gate_reason,
                has_documents=bool(rag_docs) or bool(web_docs),
            )
            tools_executed.append("LLM(blocked)")
            result = {
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
            # Fall through to the NO_ANSWER safety net below
        else:
            # ========================================================== #
            #  Multi-source numeric verification                          #
            # ========================================================== #
            _numeric_single_source = False
            _numeric_conflict = False
            _conflict_groups: List[Dict[str, Any]] = []
            _nv_result: Optional[Dict[str, Any]] = None
            _has_usable_rag = bool(rag_docs) and not adequacy_forces_llm_only
            if detect_numeric_intent(question) and _has_usable_rag:
                _nv_result = verify_numeric_consistency(
                    question, rag_docs, rag_metas,
                )
                if _nv_result["status"] == "CONFLICT":
                    _numeric_conflict = True
                    _conflict_groups = _nv_result.get("groups", [])
                    _winning = (
                        _conflict_groups[0] if _conflict_groups else {}
                    )

                    logger.warning(
                        "[NUMERIC_CONFLICT] groups=%s selected_value=%s",
                        _conflict_groups,
                        _winning.get("value"),
                    )

                    logger.info(
                        "[NUMERIC_VERIFIER] CONFLICT detected — "
                        "forcing web escalation"
                    )
                    if not web_docs:
                        web_docs = orchestrator._retrieve_web_evidence(
                            question, company_name,
                        )
                        if web_docs:
                            for _wd in web_docs:
                                _wd_meta = _wd.get("metadata", _wd)
                                _wd_meta["trust_score"] = (
                                    SourceTrustScorer.score(_wd_meta)
                                )
                            tools_executed.append(
                                "WEB_SEARCH(numeric_conflict)"
                            )
                            logger.info(
                                "[NUMERIC_VERIFIER] Web escalation "
                                "retrieved %d documents",
                                len(web_docs),
                            )
                elif _nv_result["status"] == "SINGLE_SOURCE":
                    _numeric_single_source = True
                    logger.info(
                        "[NUMERIC_VERIFIER] SINGLE_SOURCE — "
                        "confidence will be reduced to LOW"
                    )
                # CONSISTENT → normal flow, no action needed

            # ========================================================== #
            #  Answer generation                                          #
            # ========================================================== #
            usable_evidence = (
                (bool(rag_docs) and not adequacy_forces_llm_only)
                or bool(web_docs)
            )

            if usable_evidence:
                fusion_rag = rag_docs if not adequacy_forces_llm_only else []
                fusion_metas = rag_metas if not adequacy_forces_llm_only else []
                fusion_result = EvidenceFusion.fuse_evidence(
                    fusion_rag, fusion_metas, web_docs, question
                )
                fused_context = fusion_result.get("fused_context", "")

                set_llm_cost_stage("answer_generation")
                if fused_context:
                    answer = orchestrator.rag_chain.invoke(
                        {"question": question, "context": fused_context},
                        config={"callbacks": [cost_tracker_callback]},
                    )
                else:
                    context_parts: List[str] = []
                    for doc, meta in zip(fusion_rag, fusion_metas):
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
                logger.info(
                    "[POLICY_ROUTER] LLM generated answer from evidence"
                )
                tools_executed.append("LLM")
            else:
                _llm_reason = (
                    "adequacy_llm_only"
                    if adequacy_forces_llm_only
                    else "no_evidence_policy"
                )
                MetricsRecorder.record_llm_fallback(
                    question, reason=_llm_reason
                )
                set_llm_cost_stage("answer_generation")
                answer = orchestrator.llm_only_chain.invoke(
                    {"question": question},
                    config={"callbacks": [cost_tracker_callback]},
                )
                if temporal_query:
                    _TEMPORAL_DISCLAIMER = (
                        "\n\n---\n⚠️ Real-time data could not be verified "
                        "via live sources at the time of the request."
                    )
                    answer = answer.rstrip() + _TEMPORAL_DISCLAIMER
                    logger.info(
                        "[POLICY_ROUTER] LLM answer with temporal disclaimer"
                    )
                else:
                    logger.info(
                        "[POLICY_ROUTER] LLM generated answer (no evidence)"
                    )
                tools_executed.append("LLM")

            # ========================================================== #
            #  Execution state (deterministic routing contract)           #
            # ========================================================== #
            _usable_rag_count = (
                len(rag_docs) if not adequacy_forces_llm_only else 0
            )
            execution_state = ExecutionState(
                retrieval_count=_usable_rag_count,
                web_count=len(web_docs),
                adequacy_decision=(
                    adequacy_result["decision"]
                    if adequacy_result
                    else "N/A"
                ),
                numeric_intent=detect_numeric_intent(question),
                requested_year=extract_fiscal_year(question),
            )
            execution_state.resolve_mode()

            execution_path = execution_state.execution_path
            answer_type = execution_state.answer_type
            has_evidence = bool(rag_docs) or bool(web_docs)

            logger.info(
                "execution_contract", extra=execution_state.to_log_dict()
            )
            logger.info(
                "[POLICY_ROUTER] execution_state resolved: %s  "
                "(legacy execution_path=%s)",
                execution_state,
                execution_path,
            )

            # --- Stronger disclaimer for numeric LLM-only answers ---
            if (
                execution_state.final_mode == "LLM_ONLY"
                and execution_state.numeric_intent
            ):
                _FINANCIAL_DISCLAIMER = (
                    "\n\n---\n⚠️ **Note:** This answer is based on general "
                    "knowledge and may not reflect audited or finalized "
                    "financial figures. For authoritative values, please "
                    "consult the company's official investor relations "
                    "filings or regulatory disclosures."
                )
                answer = answer.rstrip() + _FINANCIAL_DISCLAIMER
                logger.info(
                    "[POLICY_ROUTER] Appended financial disclaimer "
                    "(LLM_ONLY + numeric intent)"
                )

            # --- Normalize sources ---
            formatted_sources = _normalize_sources(
                rag_metas, web_docs, execution_path
            )

            # --- Calibrated confidence signal ---
            # Combines cross-encoder relevance, numeric trust weight,
            # and routing probability into a single [0, 1] signal.
            _max_cross = float(
                retrieval_metrics.get("best_cross_score", 0.0)
            )
            _nv_trust_raw = float(
                _nv_result.get("trust_weight_sum", 0.0)
                if _nv_result is not None
                else 0.0
            )
            _nv_trust_norm = min(1.0, _nv_trust_raw)
            _routing_prob = 0.0
            if adequacy_result and adequacy_result.get("probabilities"):
                _sel_decision = adequacy_result.get("decision", "")
                _routing_prob = float(
                    adequacy_result["probabilities"].get(
                        _sel_decision, 0.0
                    )
                )
            _confidence_signal = (
                0.4 * _max_cross
                + 0.3 * _nv_trust_norm
                + 0.3 * _routing_prob
            )

            if _confidence_signal >= 0.80:
                confidence_level = "HIGH"
            elif _confidence_signal >= 0.60:
                confidence_level = "MEDIUM"
            else:
                confidence_level = "LOW"

            # Derive disclosure from base path (non-overridden).
            _, disclosure_note = _derive_confidence_and_disclosure(
                execution_path
            )

            logger.info(
                "[CONFIDENCE] signal=%.4f mapped=%s "
                "(cross=%.3f nv_trust=%.3f routing_prob=%.3f)",
                _confidence_signal,
                confidence_level,
                _max_cross,
                _nv_trust_norm,
                _routing_prob,
            )

            # --- Overrides (must stay after calibration) ---
            if _numeric_conflict:
                confidence_level = "LOW"
                disclosure_note = (
                    "Multiple conflicting financial figures were "
                    "detected across sources. The system prioritised "
                    "the value supported by the highest-trust sources."
                )
                logger.info(
                    "[CONFIDENCE] downgraded_due_to_conflict=True"
                )

            if _entity_mismatch:
                confidence_level = "LOW"
                disclosure_note = (
                    "Answer may not match the exact requested company. "
                    "Verify against official filings."
                )
                logger.info(
                    "[GUARDRAIL] Confidence reduced to LOW "
                    "due to entity mismatch"
                )

            if _numeric_single_source and confidence_level != "LOW":
                confidence_level = "LOW"
                logger.info(
                    "[NUMERIC_VERIFIER] Confidence reduced to LOW — "
                    "numeric value from single source only"
                )

            # --- Audit metadata ---
            audit_meta = _build_audit_meta(
                question, execution_path, tools_executed, web_docs
            )
            if audit_meta is not None and adequacy_result is not None:
                audit_meta["answerability"] = {
                    "decision": adequacy_result["decision"],
                    "probabilities": adequacy_result.get("probabilities", {}),
                    "logits": adequacy_result.get("logits", {}),
                    "features": adequacy_result.get("features", {}),
                    "model_version": adequacy_result.get(
                        "model_version", "unknown"
                    ),
                }

            # --- Trust analysis (audit extension) ---
            _all_trust = [
                m.get("trust_score", 0.0) for m in rag_metas
            ] + [
                wd.get("metadata", wd).get("trust_score", 0.0)
                for wd in web_docs
            ]
            if audit_meta is not None:
                audit_meta["trust_analysis"] = {
                    "average_trust": round(
                        sum(_all_trust) / len(_all_trust), 4
                    ) if _all_trust else 0.0,
                    "max_trust": round(
                        max(_all_trust), 4
                    ) if _all_trust else 0.0,
                    "numeric_trust_weight": round(_nv_trust_raw, 4),
                    "confidence_signal": round(_confidence_signal, 4),
                }

            # --- Numeric conflict audit (transparent resolution) ---
            if audit_meta is not None and _numeric_conflict:
                _winning = (
                    _conflict_groups[0] if _conflict_groups else {}
                )
                audit_meta["numeric_conflict"] = {
                    "groups": _conflict_groups,
                    "selected_value": _winning.get("value"),
                    "selected_trust_weight": _winning.get(
                        "trust_weight_sum"
                    ),
                }

            # --- Structured observability summary ---
            _extra: Dict[str, Any] = {
                "event_type": "policy_execution_summary",
                "execution_policy": execution_policy,
                "execution_path": execution_path,
                "final_mode": execution_state.final_mode,
                "tools_executed": tools_executed,
                "rag_doc_count": len(rag_docs),
                "web_doc_count": len(web_docs),
                "answer_type": answer_type,
                "coverage": has_internal,
            }
            _rid = get_request_id()
            if _rid is not None:
                _extra["request_id"] = _rid
            logger.info("policy execution complete", extra=_extra)

            from rag.telemetry import cost_tracker

            telemetry = cost_tracker.summary()
            logger.info(
                "llm_cost_summary",
                extra={
                    "event_type": "llm_cost_summary",
                    "total_input_tokens": telemetry["total_input_tokens"],
                    "total_output_tokens": telemetry["total_output_tokens"],
                    "total_cost_usd": round(
                        telemetry["total_cost_usd"], 6
                    ),
                    "stages": telemetry["stages"],
                },
            )

            # --- Build final result ---
            result = {
                "answer": answer,
                "answer_type": answer_type,
                "sources": formatted_sources,
                "confidence_level": confidence_level,
                "disclosure_note": disclosure_note,
            }
            if audit_meta is not None:
                result["_meta"] = audit_meta
            if has_evidence:
                MetricsRecorder.record_rag_answer(
                    question, has_documents=True
                )

        # -------------------------------------------------------------- #
        # FINAL SAFETY NET: System must ALWAYS produce an answer.         #
        # If the pipeline returned NO_ANSWER (e.g. guardrail block on    #
        # year/metric mismatch), fall back to a direct LLM call.          #
        # -------------------------------------------------------------- #
        if result.get("answer_type") == "NO_ANSWER":
            logger.warning(
                "[FALLBACK] Pipeline returned NO_ANSWER — invoking "
                "LLM fallback to guarantee output"
            )
            try:
                set_llm_cost_stage("answer_generation")
                fallback_answer = orchestrator.llm_only_chain.invoke(
                    {"question": question},
                    config={"callbacks": [cost_tracker_callback]},
                )
                result = {
                    "answer": str(fallback_answer),
                    "answer_type": "LLM_FALLBACK",
                    "sources": [],
                    "confidence_level": "LOW",
                    "disclosure_note": (
                        "Generated without verified internal sources."
                    ),
                }
                MetricsRecorder.record_llm_fallback(
                    question, reason="no_answer_safety_net"
                )
                logger.info("[FALLBACK] LLM fallback answer generated")
            except Exception as fb_exc:
                logger.error(
                    "[FALLBACK] LLM fallback also failed: %s", fb_exc
                )
                result = {
                    "answer": (
                        "I was unable to find verified documents for this "
                        "query, and the general-knowledge fallback also "
                        "encountered an error. Please try rephrasing."
                    ),
                    "answer_type": "LLM_FALLBACK",
                    "sources": [],
                    "confidence_level": "LOW",
                    "disclosure_note": (
                        "Generated without verified internal sources."
                    ),
                }

        return result

    except RuntimeError as e:
        error_msg = str(e)
        logger.error("[POLICY_ROUTER] RuntimeError: %s", error_msg)

        if "ChromaDB collection is EMPTY" in error_msg or "FATAL ERROR" in error_msg:
            return {
                "answer": (
                    "The knowledge base is not available. Please ensure "
                    "documents have been ingested by running the ingestion process."
                ),
                "answer_type": "sagealpha_rag",
                "sources": [],
            }
        if "CHROMA_API_KEY" in error_msg or "environment variable" in error_msg:
            return {
                "answer": (
                    "Service configuration error. Please check that all "
                    "required environment variables are set correctly."
                ),
                "answer_type": "sagealpha_rag",
                "sources": [],
            }
        return {
            "answer": (
                f"I apologize, but I encountered an error while initializing "
                f"the system: {error_msg}. Please contact support."
            ),
            "answer_type": "sagealpha_rag",
            "sources": [],
        }

    except Exception as exc:
        logger.exception(
            "[POLICY_ROUTER] HARD FAILURE — policy execution aborted "
            "(%s: %s)",
            type(exc).__name__,
            exc,
        )
        return {
            "answer": (
                "I apologize, but I encountered an error while processing "
                "your query. Please try again."
            ),
            "answer_type": "sagealpha_rag",
            "sources": [],
        }
