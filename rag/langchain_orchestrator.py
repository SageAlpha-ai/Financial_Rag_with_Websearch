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

import logging
import os
import re
from typing import Dict, Any, List, Optional, Tuple
import json

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from rank_bm25 import BM25Okapi

from config.settings import get_config
from vectorstore.chroma_client import get_collection

# Web search imports
from rag.web_search import WebSearchEngine, should_trigger_web_search
from rag.investor_scraper import InvestorRelationsScraper
from rag.document_extractor import DocumentExtractor
from rag.temp_storage import get_temp_storage
from rag.evidence_fusion import EvidenceFusion
from rag.source_formatter import SourceFormatter
from rag.company_validator import CompanyValidator
from rag.query_classifier import QueryClassifier
from rag.intent_extractor import IntentExtractor

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


def _extract_entity_from_query(query: str) -> Optional[str]:
    """Extracts company/entity from query."""
    query_lower = query.lower()
    
    entity_mappings = {
        # Generic variations can be added here if needed, but avoiding hardcoded business entities
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


def _validate_answerability(
    query: str,
    documents: List[str],
    metadatas: List[Dict]
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    OpenAI-style answerability validation.
    
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
    requested_entity = _extract_entity_from_query(query)
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
        
        import chromadb
        logger.info(f"[CHROMA_VERSION] {chromadb.__version__}")
        
        # Azure OpenAI LLM
        self.llm = AzureChatOpenAI(
            azure_endpoint=config.azure_openai.endpoint,
            azure_deployment=config.azure_openai.chat_deployment,
            api_key=config.azure_openai.api_key,
            api_version=config.azure_openai.api_version,
            temperature=0.0,
        )
        
        # Initialize Planner LLM (if enabled)
        self.planner_llm = None
        if config.enable_query_planner and config.azure_openai.planner_deployment:
            try:
                self.planner_llm = AzureChatOpenAI(
                    azure_endpoint=config.azure_openai.endpoint,
                    azure_deployment=config.azure_openai.planner_deployment,
                    api_key=config.azure_openai.api_key,
                    api_version=config.azure_openai.api_version,
                    temperature=0.0,
                )
                logger.info(f"[ORCHESTRATOR] Planner LLM initialized: {config.azure_openai.planner_deployment}")
            except Exception as e:
                logger.warning(f"[ORCHESTRATOR] Failed to initialize Planner LLM: {e}")
        
        # Azure OpenAI Embeddings (MUST match ingestion model)
        # For text-embedding-3-large, don't pass model parameter (deployment name is sufficient)
        embedding_kwargs = {
            "azure_endpoint": config.azure_openai.endpoint,
            "azure_deployment": config.azure_openai.embeddings_deployment,
            "api_key": config.azure_openai.api_key,
            "api_version": config.azure_openai.api_version,
        }
        

        
        self.embeddings = AzureOpenAIEmbeddings(**embedding_kwargs)
        
        self.output_parser = StrOutputParser()
        
        # Get Chroma collections
        # Dynamically load ALL collections from the active Chroma database
        from vectorstore.chroma_client import get_all_collections

        try:
            self.collections = get_all_collections()
            self.chroma_available = len(self.collections) > 0
            logger.info(f"[ORCHESTRATOR] Dynamically loaded {len(self.collections)} collections")
        except Exception as e:
            logger.warning(f"[ORCHESTRATOR] Failed to load collections: {e}")
            self.collections = []
            self.chroma_available = False
        
        # VERIFY ChromaDB is not empty (WARN but don't fail - allow LLM fallback)
        if self.chroma_available and self.collections:
            total_docs = 0
            for col in self.collections:
                try:
                    doc_count = col.count()
                    total_docs += doc_count
                    logger.info(f"Collection '{col.name}': {doc_count} documents")
                except Exception as e:
                    logger.warning(f"[ORCHESTRATOR] Failed to count docs in {col.name}: {e}")
            
            if total_docs == 0:
                warning_msg = (
                    "WARNING: All ChromaDB collections are EMPTY. "
                    "No documents have been ingested. "
                    "RAG queries will fall back to LLM-only answers. "
                    "Ensure documents are ingested via the external data pipeline."
                )
                logger.warning("=" * 60)
                logger.warning(warning_msg)
                logger.warning("=" * 60)
        else:
            logger.warning("[ORCHESTRATOR] Skipping collection status check (Chroma unavailable)")
            
        logger.info("=" * 60)
        
        logger.info("=" * 60)
        
        # Initialize hybrid retrieval
        self._setup_retrievers()
        
        # Initialize web search components
        self.web_search = WebSearchEngine()
        self.investor_scraper = InvestorRelationsScraper()
        self.document_extractor = DocumentExtractor()
        self.temp_storage = get_temp_storage()
        self._current_investor_metadata = {}  # Store investor relations metadata for current query
        
        # Setup prompts and chains
        self._setup_chains()

        # Initialize Intent Extractor (Phase 1: Observation Mode)
        try:
             self.intent_extractor = IntentExtractor()
             logger.info("[ORCHESTRATOR] IntentExtractor initialized")
        except Exception as e:
             logger.warning(f"[ORCHESTRATOR] Failed to initialize IntentExtractor: {e}")
             self.intent_extractor = None
    
    def _setup_retrievers(self):
        """Setup BM25 index from Chroma documents."""
        try:
            logger.info("Loading documents from Chroma for BM25 indexing...")
            all_documents = []
            all_metadatas = []
            
            for col in self.collections:
                docs, metas = _load_all_documents_from_chroma(col)
                all_documents.extend(docs)
                all_metadatas.extend(metas)
            
            if not all_documents:
                logger.warning("No documents loaded from Chroma. BM25 will be disabled.")
                self.bm25_index = None
                return
            
            logger.info(f"Initializing BM25 index with {len(all_documents)} documents...")
            self.bm25_index = BM25Index(all_documents, all_metadatas)
            logger.info("BM25 index initialized successfully")
        except Exception as e:
            logger.error(f"Failed to setup BM25 index: {e}", exc_info=True)
            self.bm25_index = None
    
    def _build_company_filter(self, intent: Any) -> Optional[Dict[str, Any]]:
        """
        Build strict metadata filter using exact match.
        Chroma v2 safe.
        """
        if not intent or not intent.company:
            logger.info("[METADATA_FILTER] None (no company detected)")
            return None

        company = intent.company.strip()
        logger.info(f"[METADATA_FILTER] Exact match filter for company: {company}")

        return {
            "$or": [
                {"company_name": {"$eq": company}},
                {"company": {"$eq": company}}
            ]
        }

    def _setup_chains(self):
        """Setup LangChain prompts and LCEL chains."""
        
        # RAG prompt (Azure-safe, strict document-based answers)
        self.rag_template = """You are a financial compliance assistant.

Answer the question using only the information provided in the context below.

Context documents:
{context}

Question: {question}

Guidelines:
- Use information from the context documents to answer the question.
- Include specific numbers, dates, and facts exactly as they appear in the context.
- Preserve exact numeric values without modification.
- If the answer is explicitly present in the context, state it clearly.
- If the answer is not present in the context, respond exactly with: "This information is not available in the internal documents."
- Cite sources when referencing specific data from the context.
- For web documents, cite the official company website URL.
- For internal documents, cite the document name and page number if available.

Answer:"""
        
        self.rag_prompt = ChatPromptTemplate.from_template(self.rag_template)
        self.rag_chain = self.rag_prompt | self.llm | self.output_parser
        
        # LLM-only prompt (Azure-safe, general knowledge)
        self.llm_only_template = """You are a general financial assistant.

Respond naturally and conversationally to the user's question.

Question: {question}

Guidelines:
- Use your general knowledge to provide helpful information.
- Respond in a clear and educational manner.
- If exact figures or current values are unavailable, state that values may be approximate.
- Provide educational information only.
- Do not give personalized or binding financial advice.

Answer:"""
        
        self.llm_only_prompt = ChatPromptTemplate.from_template(self.llm_only_template)
        self.llm_only_chain = self.llm_only_prompt | self.llm | self.output_parser
    
    def _retrieve_documents_hybrid(self, question: str) -> Tuple[List[str], List[Dict]]:
        """
        STRICT retrieval: ALWAYS attempts to retrieve documents from ChromaDB.
        
        Uses SAME embedding model as ingestion (must match deployment name).
        Supports fiscal year filtering when query specifies a year.
        """
        if not getattr(self, "chroma_available", False):
            return [], []

        try:
            config = get_config()
            logger.info("[RETRIEVER] Embedding query using Azure OpenAI")
            logger.info(f"[RETRIEVER] Embedding model: {config.azure_openai.embeddings_deployment}")
            
            # Extract fiscal year from query
            requested_year = _extract_fiscal_year(question)
            if requested_year:
                logger.info(f"[RETRIEVER] Detected fiscal year in query: {requested_year}")
            
            # Generate embeddings using SAME model as ingestion
            query_embedding = self.embeddings.embed_query(question)
            logger.info(f"[RETRIEVER] Query embedding generated (dimension: {len(query_embedding)})")
            
            logger.info("[RETRIEVER] Searching ChromaDB using cosine similarity")
            
            # Query all collections and merge results
            all_chroma_docs = []
            all_chroma_metas = []
            all_chroma_distances = [] # Keep track for logging only
            
            for col in self.collections:
                # If fiscal year specified, try year-filtered retrieval first
                year_docs = []
                year_metas = []
                if requested_year:
                    try:
                        year_results = col.query(
                            query_embeddings=[query_embedding],
                            n_results=5,
                            where={"fiscal_year": requested_year},
                            include=["documents", "metadatas", "distances"]
                        )
                        year_docs = year_results.get("documents", [[]])[0] if year_results.get("documents") else []
                        year_metas = year_results.get("metadatas", [[]])[0] if year_results.get("metadatas") else []
                    except Exception as e:
                        logger.warning(f"[RETRIEVER] Year-filtered retrieval failed for {col.name}: {e}")
                
                # Always perform general retrieval
                chroma_results = col.query(
                    query_embeddings=[query_embedding],
                    n_results=5,
                    include=["documents", "metadatas", "distances"]
                )
                
                chroma_docs = chroma_results.get("documents", [[]])[0] if chroma_results.get("documents") else []
                chroma_metas = chroma_results.get("metadatas", [[]])[0] if chroma_results.get("metadatas") else []
                chroma_distances = chroma_results.get("distances", [[]])[0] if chroma_results.get("distances") else []
                
                if chroma_distances:
                    all_chroma_distances.extend(chroma_distances)

                # Combine results: year-specific first, then general (deduplicated)
                if year_docs:
                    seen_texts = {doc.strip().lower() for doc in year_docs}
                    additional_docs = []
                    additional_metas = []
                    for doc, meta in zip(chroma_docs, chroma_metas):
                        if doc.strip().lower() not in seen_texts:
                            additional_docs.append(doc)
                            additional_metas.append(meta)
                    
                    merged_col_docs = year_docs + additional_docs
                    merged_col_metas = year_metas + additional_metas
                else:
                    merged_col_docs = chroma_docs
                    merged_col_metas = chroma_metas
                
                all_chroma_docs.extend(merged_col_docs)
                all_chroma_metas.extend(merged_col_metas)
            
            chroma_docs = all_chroma_docs
            chroma_metas = all_chroma_metas
            
            if not chroma_docs:
                logger.warning("[RETRIEVER] No documents retrieved from any ChromaDB collection")
            else:
                 logger.info(f"[RETRIEVER] Retrieved {len(chroma_docs)} total documents from {len(self.collections)} collections")

            
            # BM25 retrieval if numeric intent detected
            bm25_docs = []
            bm25_metas = []
            if _detect_numeric_intent(question) and self.bm25_index is not None:
                bm25_docs, bm25_metas = self.bm25_index.search(question, top_k=5)
                logger.info(f"[RETRIEVER] BM25 retrieved {len(bm25_docs)} documents")
            
            # Merge and deduplicate results
            merged_docs, merged_metas = _deduplicate_documents(
                chroma_docs, chroma_metas,
                bm25_docs, bm25_metas
            )
            
            logger.info(f"[RETRIEVER] Total documents retrieved: {len(merged_docs)}")
            
            # Log first document preview if available
            if merged_docs:
                first_doc_preview = merged_docs[0][:500]
                logger.info(f"[RETRIEVER] First document preview: {first_doc_preview}...")
            
            return merged_docs, merged_metas
        except Exception as e:
            logger.error(f"[RETRIEVER] Hybrid document retrieval failed: {e}", exc_info=True)
            return [], []
    
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
            # Extract company name if not provided
            if not company_name:
                company_name = _extract_entity_from_query(question)
            
            if not company_name:
                logger.info("[WEB_SEARCH] No company name found in query, using full query for general search")
                # Fallback to general search if no company extracted
                search_results = self.web_search.search_general(question)
                return self._process_general_search_results(search_results)
            
            logger.info(f"[WEB_SEARCH] Searching for company: {company_name}")
            
            # Search for investor relations page
            search_result = self.web_search.search_company_investor_relations(company_name)
            
            if not search_result.get("success"):
                logger.warning(f"[WEB_SEARCH] Could not find investor relations page: {search_result.get('error')}")
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
                "company": company_name
            }
            
            # Scrape investor page
            logger.info(f"[WEB_SEARCH] Scraping investor page: {investor_url}")
            scrape_result = self.investor_scraper.scrape_investor_page(investor_url)
            
            if not scrape_result.get("success"):
                logger.warning(f"[WEB_SEARCH] Failed to scrape investor page: {scrape_result.get('error')}")
                return []
            
            # Download and extract documents
            web_documents = []
            documents_found = scrape_result.get("documents", [])
            
            # Prioritize annual reports and financial statements
            priority_types = ["annual_report", "financial_statement", "10-k", "10-q", "earnings_release"]
            priority_docs = [d for d in documents_found if d.get("type") in priority_types]
            other_docs = [d for d in documents_found if d.get("type") not in priority_types]
            
            # Process priority documents first (limit to 3 to avoid too many downloads)
            for doc_info in (priority_docs + other_docs)[:3]:
                doc_url = doc_info.get("url")
                if not doc_url:
                    continue
                
                try:
                    logger.info(f"[WEB_SEARCH] Downloading document: {doc_info.get('title', 'Unknown')}")
                    file_path, download_meta = self.investor_scraper.download_document(doc_url)
                    
                    if not file_path:
                        continue
                    
                    # Extract text
                    extract_result = self.document_extractor.extract_text(file_path, download_meta)
                    
                    if extract_result.get("success"):
                        # Store in temporary storage with investor relations metadata
                        storage_meta = self.temp_storage.store_document(
                            file_path,
                            {
                                **download_meta,
                                "title": doc_info.get("title", ""),
                                "type": doc_info.get("type", ""),
                                "url": doc_url,
                                "investor_url": getattr(self, '_current_investor_metadata', {}).get("investor_url"),
                                "investor_root": getattr(self, '_current_investor_metadata', {}).get("investor_url"),
                                "official_domain": getattr(self, '_current_investor_metadata', {}).get("official_domain"),
                                "company": getattr(self, '_current_investor_metadata', {}).get("company")
                            }
                        )
                        
                        web_documents.append({
                            "text": extract_result.get("text", ""),
                            "metadata": {
                                **storage_meta,
                                "title": doc_info.get("title", ""),
                                "url": doc_url,
                                "type": doc_info.get("type", "")
                            }
                        })
                        
                        logger.info(f"[WEB_SEARCH] Extracted {extract_result.get('pages', 0)} pages from document")
                    
                except Exception as e:
                    logger.error(f"[WEB_SEARCH] Failed to process document {doc_url}: {e}")
                    continue
            
            logger.info(f"[WEB_SEARCH] Retrieved {len(web_documents)} web documents")
            return web_documents
            
        except Exception as e:
            logger.error(f"[WEB_SEARCH] Web evidence retrieval failed: {e}", exc_info=True)
            return []

    def _process_general_search_results(self, search_results: Dict[str, Any]) -> List[Dict]:
        """Process results from general web search."""
        if not search_results.get("success"):
            logger.warning(f"[WEB_SEARCH] General search failed: {search_results.get('error')}")
            return []
        
        web_documents = []
        for result in search_results.get("search_results", []):
            web_documents.append({
                "text": result.get("snippet", ""),
                "metadata": {
                    "source": "web_search",
                    "title": result.get("title", ""),
                    "url": result.get("link", ""),
                    "type": "general_web"
                }
            })
        return web_documents

    def _probe_internal_knowledge(self, question: str) -> List[Dict]:
        """
        Lightweight Semantic Probe.
        Checks if ChromaDB contains relevant documents for the query.
        Returns top 3 documents if found, else empty list.
        """
        try:
            # Generate embedding
            query_embedding = self.embeddings.embed_query(question)
            
            # Fast query (using default collection)
            results_list = []
            for col in self.collections:
                results_list.append(col.query(
                    query_embeddings=[query_embedding],
                    n_results=3,
                    include=["documents", "metadatas", "distances"]
                ))
            
            # Determine success if ANY collection returned results
            # For simplicity in this probe, we just check if ANYTHING was found
            
            docs = []
            metas = []
            distances = []
            
            for results in results_list:
                d = results.get("documents", [[]])[0] if results.get("documents") else []
                m = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
                dst = results.get("distances", [[]])[0] if results.get("distances") else []
                docs.extend(d)
                metas.extend(m)
                distances.extend(dst)
            
            # Filter by relevance threshold (e.g., distance < 1.0 for cosine distance)
            # Only count as "Internal Knowledge" if reasonably close
            relevant_docs = []
            
            for i, dist in enumerate(distances):
                 # Threshold: 1.5 is generous for cosine distance (usually 0 to 2) 
                 # Adjust based on your embedding model. 
                 # For ada-002, < 0.3-0.4 is very close, < 0.5 is relevant.
                 # Conservatively, we just check if ANY result was returned.
                 relevant_docs.append({
                     "content": docs[i],
                     "metadata": metas[i],
                     "distance": dist
                 })
            
            return relevant_docs

        except Exception as e:
            logger.warning(f"[PROBE] Failed to probe internal knowledge: {e}")
            return []

    def _determine_mode(self, intent: Any, internal_doc_count: int, best_distance: Optional[float]) -> str:
        """
        Deterministic mode selection based on strict Policy Matrix.
        
        POLICY:
        1) definition: None company -> llm; Else -> (strong_internal -> rag) elif requires_web -> web else llm
        2) historical_data: strong_internal -> rag else web
        3) realtime_data: requires_realtime -> web else llm
        4) comparison: strong_internal & requires_web -> hybrid; strong_internal -> rag; else web
        5) unknown: None company -> llm; strong_internal -> rag else web
        """
        # Default fallback
        if not intent:
            if internal_doc_count > 0 and best_distance is not None and best_distance < 0.55:
                return "rag"
            else:
                return "llm"

        strong_internal = (
            internal_doc_count > 0 and 
            best_distance is not None and 
            best_distance < 0.55
        )
        
        mode = "llm" # Default safety
        
        # Policy 1: Definition
        if intent.query_type == "definition":
            if not intent.company:
                mode = "llm"
            else:
                if strong_internal:
                    mode = "rag"
                elif intent.requires_web:
                    mode = "web"
                else:
                    mode = "llm"

        # Policy 2: Historical Data
        elif intent.query_type == "historical_data":
            if strong_internal:
                mode = "rag"
            else:
                mode = "web"

        # Policy 3: Realtime Data
        elif intent.query_type == "realtime_data":
            if intent.requires_realtime:
                mode = "web"
            else:
                mode = "llm"

        # Policy 4: Comparison
        elif intent.query_type == "comparison":
            if strong_internal and intent.requires_web:
                mode = "hybrid"
            elif strong_internal:
                mode = "rag"
            else:
                mode = "web"

        # Policy 5: Unknown / General
        else: # includes 'unknown' and 'report' (treat report as unknown/hybrid dependent on signals)
            # Special case for explicit 'report' type if not handled elsewhere, 
            # usually reports fall into historical/comparison, but if 'report':
            if intent.query_type == "report":
                 if strong_internal and intent.requires_web:
                     mode = "hybrid"
                 elif strong_internal:
                     mode = "rag"
                 else:
                     mode = "web"
            elif not intent.company:
                mode = "llm"
            else:
                if strong_internal:
                    mode = "rag"
                else:
                    mode = "web"

        logger.info(f"[ROUTER_POLICY] query_type={intent.query_type}, company={intent.company}, strong_internal={strong_internal}")
        return mode

    def _plan_query(
        self,
        question: str,
        best_distance: Optional[float],
        internal_doc_count: int,
        internal_docs_found: bool
    ) -> Dict[str, Any]:
        """
        Planner Phase (gpt-4o-mini).
        
        Decides routing strategy based on semantic signals and query intent.
        
        Signals:
        - best_semantic_distance: Lower is better (0.0 = exact match). < 0.3 is very strong. > 0.5 is weak.
        - internal_docs_found: Whether any documents were retrieved.
        """
        default_plan = {"mode": "rag", "reason": "Default fallback"}
        
        try:
            if not self.planner_llm:
                 # Fallback if no planner model configured
                 if internal_docs_found:
                     return {"mode": "rag", "reason": "Fallback: Internal docs found, planner disabled"}
                 else:
                     return {"mode": "llm", "reason": "Fallback: No docs found, planner disabled"}

            planner_prompt = f"""You are an intelligent AI routing engine.

You must decide how to answer a query based on semantic retrieval signals.

Signals provided:
- Query: "{question}"
- Internal Documents Found: {internal_docs_found}
- Count: {internal_doc_count}
- Best Semantic Distance: {best_distance if best_distance is not None else "N/A"}
  (Note: Distance 0.0 is exact match. < 0.4 is strong. > 0.5 is weak.)

Available modes:
- "rag"     → Use internal knowledge only. (Data-dependent query requiring specific internal evidence)
- "web"     → Use web search only. (Real-time, news, stock price, today's data)
- "hybrid"  → Use both internal knowledge and web. (Mixed intent, weak internal match but relevant)
- "llm"     → Use model knowledge only. (General definition, explanation, greetings, no internal match)

Important Reasoning Guidelines:

1. CONCEPTUAL / DEFINITION QUERIES:
   - If the query asks for a general explanation, definition, or concept (e.g., "What is EBITDA?", "Explain inflation", "Define revenue"), prefer "llm".
   - Do NOT choose "rag" just because internal documents mention the term.
   - Only choose "rag" if the question asks for SPECIFIC internal data (e.g., "What was FY2023 revenue?", "Show EBITDA margin").

2. DATA-DEPENDENT QUERIES:
   - If the query requires historical financial figures, company-specific metrics, or document-based evidence, prefer "rag".
   - Strong semantic similarity is a positive signal here.

3. REAL-TIME QUERIES:
   - If query asks for "today", "current", "latest", "stock price", prefer "web" or "hybrid".

4. MIXED:
   - If query compares historical data with current market data, prefer "hybrid".

5. FALLBACK:
   - If no internal documents found or distance > 0.8, prefer "llm" (or "web").

Return STRICT JSON ONLY:
{{
  "mode": "rag" | "web" | "hybrid" | "llm",
  "reason": "short explanation"
}}
"""
            response = self.planner_llm.invoke(planner_prompt)
            
            # Helper to clean response content
            content = response.content.replace("```json", "").replace("```", "").strip()
            
            import json
            plan = json.loads(content)
            
            mode = plan.get("mode", "rag").lower()
            if mode not in ["rag", "web", "hybrid", "llm"]:
                mode = "rag"
            
            return {
                "mode": mode,
                "reason": plan.get("reason", "No reason provided")
            }

        except Exception as e:
            logger.warning(f"[PLANNER] Failed. Using safe fallback. Error: {e}")
            
            # Smart Fallback logic
            if internal_docs_found:
                return {"mode": "rag", "reason": "Fallback: Internal knowledge detected"}
            else:
                return {"mode": "llm", "reason": "Fallback: No internal knowledge detected"}

    def answer_query(self, question: str) -> Dict[str, Any]:
        """
        Deterministic Orchestrator.
        
        Routing Logic:
        1. Company Check (Supported vs Unsupported)
        2. If Supported -> Semantic Search (ChromaDB)
           - Strong Match (Similarity >= threshold) -> RAG Only
           - Weak/No Match -> Hybrid (Web + LLM)
        3. If Unsupported -> Web + LLM
        """
        try:
            config = get_config()
            logger.info("[QUERY] Processing query")
            logger.info(f"[QUERY] Query text: {question}")
            
            question_lower = question.lower()
            
            # Initialize return variables
            intent = None
            fused_context = ""

            # PHASE 0: INTENT EXTRACTION (Log Only)
            if self.intent_extractor:
                try:
                    intent = self.intent_extractor.extract(question)
                    logger.info(f"[INTENT] {intent.dict()}")
                except Exception as e:
                    logger.warning(f"[INTENT] Extraction failed: {e}")
            
            # Initialize evidence buckets
            rag_docs = []
            rag_metas = []
            distances = []
            web_docs = []
            
            # PHASE 1: SEMANTIC PROBE
            logger.info("[ROUTER] Phase 1: Semantic Probe across ALL collections")
            
            best_distance = None
            internal_doc_count = 0
            internal_docs_found = False
            
            try:
                query_embedding = self.embeddings.embed_query(question)
                
                raw_docs = []
                raw_metas = []
                raw_distances = []
                
                # Build metadata filter for company isolation
                where_filter = self._build_company_filter(intent)
                
                for col in self.collections:
                    try:
                        if where_filter:
                            logger.info(f"[COLLECTION_QUERY] Using filter for collection: {col.name}")
                            results = col.query(
                                query_embeddings=[query_embedding],
                                n_results=5,
                                where=where_filter,
                                include=["documents", "metadatas", "distances"]
                            )
                        else:
                            results = col.query(
                                query_embeddings=[query_embedding],
                                n_results=5,
                                include=["documents", "metadatas", "distances"]
                            )
                        
                        d = results.get("documents", [[]])[0] if results.get("documents") else []
                        m = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
                        dist = results.get("distances", [[]])[0] if results.get("distances") else []

                        if m:
                            logger.info(f"[DEBUG_SAMPLE_METADATA] {m[0]}")
                        
                        raw_docs.extend(d)
                        raw_metas.extend(m)
                        raw_distances.extend(dist)

                    except Exception as e:
                        logger.error(f"[PROBE_ERROR] Failed to query collection {col.name}: {e}")
                        continue
                
                internal_doc_count = len(raw_docs)
                internal_docs_found = internal_doc_count > 0
                
                if internal_docs_found and raw_distances:
                    best_distance = min(raw_distances)
                    # Store purely for potential use if RAG is chosen
                    rag_docs = raw_docs
                    rag_metas = raw_metas
                    distances = raw_distances
                
                logger.info(f"[ROUTER] Probe Results: Found {internal_doc_count} docs. Best Dist: {best_distance}")
                
            except Exception as e:
                logger.error(f"[ROUTER] Semantic probe failed: {e}")
                internal_docs_found = False
            
            # PHASE 2: DETERMINISTIC ROUTING
            logger.info("[ROUTER] Phase 2: Deterministic Routing")
            
            mode = self._determine_mode(
                intent=intent,
                internal_doc_count=internal_doc_count,
                best_distance=best_distance
            )
            
            reason = "Determined by strict routing rules based on intent and semantic probe."
            
            logger.info(f"[ROUTER] Deterministic Mode: {mode.upper()}")
            
            # PHASE 3: EXECUTION
            
            # Branch 1: LLM Only
            if mode == "llm":
                logger.info("[ROUTER] Executing LLM-only path")
                chain = get_llm_only_chain()
                answer = chain.invoke({"question": question})
                return {
                    "answer": answer,
                    "answer_type": "sagealpha_llm",
                    "sources": [],
                    "intent": intent.dict() if intent else None,
                    "context": "General Knowledge (LLM Only)"
                }
            
            # Branch 2: Web Only (Treat as Hybrid but without RAG docs)
            if mode == "web":
                logger.info("[ROUTER] Executing Web-only path")
                try:
                    web_docs = self._retrieve_web_evidence(question)
                    rag_docs = [] # Ensure empty
                    rag_metas = []
                except Exception as e:
                    logger.warning(f"[ROUTER] Web retrieval failed: {e}")
                    # Fallback to LLM if web fails
                    chain = get_llm_only_chain()
                    answer = chain.invoke({"question": question})
                    return {
                        "answer": answer,
                        "answer_type": "sagealpha_llm",
                        "sources": [],
                        "intent": intent.dict() if intent else None,
                        "context": "General Knowledge (Web Search Failed)"
                    }

            # Branch 3: RAG Only (Already have docs from probe)
            if mode == "rag":
                logger.info("[ROUTER] Executing RAG-only path")
                web_docs = [] # Ensure empty
                # rag_docs populated from probe
            
            # Branch 4: Hybrid (RAG + Web)
            if mode == "hybrid":
                logger.info("[ROUTER] Executing Hybrid path")
                try:
                    # rag_docs already populated from probe
                    web_docs = self._retrieve_web_evidence(question)
                except Exception as e:
                    logger.warning(f"[ROUTER] Web portion of hybrid failed: {e}")
            
            # Final Synthesis (Common for RAG, Web, Hybrid)
            
            # Determine answer type string for frontend
            if mode == "hybrid":
                answer_type = "sagealpha_hybrid_search"
            elif mode == "web":
                answer_type = "sagealpha_ai_search"
            elif mode == "rag":
                answer_type = "sagealpha_rag"
            else:
                answer_type = "llm_fallback"

            
            # Fuse Evidence
            fusion_result = EvidenceFusion.fuse_evidence(
                rag_docs,
                rag_metas,
                web_docs,
                question
            )
            
            fused_context = fusion_result.get("fused_context", "")
            
            # Final Synthesis
            # Final Synthesis
            system_prompt = """You are an enterprise-grade financial AI assistant.

You generate answers using three possible evidence sources:
1. Internal enterprise documents (retrieved via semantic search)
2. Real-time web search results
3. Your reasoning capability for structured synthesis

You must follow these rules strictly:

### Evidence Usage Rules
1. Use INTERNAL DOCUMENTS only for:
   - Historical financial figures
   - Company-reported revenue
   - EPS, margins, financial statements
   - Official document-based metrics

2. Use WEB SEARCH only for:
   - Current stock price
   - Live market data
   - Latest market capitalization
   - Current news

3. NEVER:
   - Estimate values using assumed P/E ratios
   - Infer stock price from EPS unless explicitly provided
   - Invent financial figures
   - Perform speculative valuation math
   - Combine unrelated web results
   - Use weak or informal sources (forums, Reddit, speculative blogs)

4. If live data is not clearly available from web sources:
   - State that current market data could not be verified
   - Do NOT fabricate estimates

### Hybrid Query Handling
If the query requires both historical and current data:
1. Present internal historical data first.
2. Present verified current data second.
3. Provide structured comparison and analytical interpretation.
4. Keep reasoning grounded in provided evidence only.

### Accuracy & Integrity Rules
- Only use numbers explicitly present in the provided context.
- If multiple web sources differ, present a reasonable range.
- Do not speculate.
- Do not project future performance unless explicitly requested.
- Do not mention RAG, Web Search, or system internals.

### Output Style
- Professional tone
- Clear section headers
- Structured formatting
- Tables only when helpful
- No emojis
- No casual phrasing
- No internal system references

CONTEXT:
{context}

Question:
{question}

Answer:"""

            generation_prompt = ChatPromptTemplate.from_messages([
                ("user", system_prompt)
            ])
            
            final_context = fused_context if fused_context else "No specific documents found. Answer based on general knowledge."
            
            chain = generation_prompt | self.llm | StrOutputParser()
            
            logger.info("[GENERATOR] Generating final answer...")
            answer = chain.invoke({
                "question": question,
                "context": final_context
            })
            
            # Format sources
            formatted_sources = SourceFormatter.format_sources(rag_metas, web_docs)
            
            return {
                "answer": answer,
                "answer_type": answer_type,
                "sources": formatted_sources if formatted_sources else [],
                "intent": intent.dict() if intent else None,
                "context": final_context
            }

        except Exception as e:
            logger.exception("LangChain orchestration failed")
            return {
                "answer": "I encountered an internal error while processing your request, but here is my best possible response based on available knowledge.",
                "answer_type": "llm_fallback",
                "sources": []
            }
    
    # _extract_sources method removed - now using SourceFormatter.format_sources()


# Singleton instance
_orchestrator: Optional[LangChainOrchestrator] = None


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
        azure_deployment=config.azure_openai.chat_deployment,
        api_key=config.azure_openai.api_key,
        api_version=config.azure_openai.api_version,
        temperature=0.0,
    )
    
    # LLM-only prompt (Azure-safe, general knowledge)
    llm_only_template = """You are a general financial assistant.

Answer the user's question using your general knowledge.
Keep responses clear and helpful.

If exact or current figures are not available, say they may be approximate.
Provide educational information only and avoid personalized financial advice.

Question: {question}

Answer:"""
    
    llm_only_prompt = ChatPromptTemplate.from_template(llm_only_template)
    llm_only_chain = llm_only_prompt | llm | StrOutputParser()
    
    return llm_only_chain


def get_orchestrator() -> LangChainOrchestrator:
    """Get singleton orchestrator instance."""
    global _orchestrator
    
    if _orchestrator is not None:
        return _orchestrator
    
    try:
        logger.info("[ORCHESTRATOR] Initializing LangChain orchestrator...")
        _orchestrator = LangChainOrchestrator()
        logger.info("[ORCHESTRATOR] Orchestrator initialized successfully")
        return _orchestrator
    except Exception as e:
        logger.warning(f"[ORCHESTRATOR] Initialization issue: {e}")
        logger.warning("[ORCHESTRATOR] Falling back to LLM-only mode")
        
        class LLMOnlyOrchestrator:
            def answer_query(self, question: str):
                from rag.langchain_orchestrator import get_llm_only_chain
                chain = get_llm_only_chain()
                answer = chain.invoke({"question": question})
                return {
                    "answer": answer,
                    "answer_type": "llm_fallback",
                    "sources": []
                }
        
        _orchestrator = LLMOnlyOrchestrator()
        return _orchestrator


def answer_query_simple(question: str) -> Dict[str, Any]:
    """
    Simplified interface for API.
    
    Uses OpenAI-style answerability validation with automatic LLM fallback.
    """
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
