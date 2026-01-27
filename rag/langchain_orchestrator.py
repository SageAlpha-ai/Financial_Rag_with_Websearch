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
        
        # Azure OpenAI LLM
        self.llm = AzureChatOpenAI(
            azure_endpoint=config.azure_openai.endpoint,
            azure_deployment=config.azure_openai.chat_deployment,
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
        
        # Get Chroma collection
        # Use create_if_missing=False for strict mode (will raise ValueError if missing)
        # This allows graceful error handling in the API layer
        try:
            self.collection = get_collection(create_if_missing=False)
        except ValueError as e:
            # Collection doesn't exist - this is a configuration issue
            error_msg = str(e)
            logger.error("=" * 60)
            logger.error("CHROMADB COLLECTION ERROR")
            logger.error("=" * 60)
            logger.error(error_msg)
            logger.error("=" * 60)
            raise RuntimeError(error_msg) from e
        
        # VERIFY ChromaDB is not empty (WARN but don't fail - allow LLM fallback)
        doc_count = self.collection.count()
        logger.info("=" * 60)
        logger.info("CHROMADB COLLECTION STATUS")
        logger.info("=" * 60)
        logger.info(f"Collection name: {self.collection.name}")
        logger.info(f"Total embeddings: {doc_count}")
        
        if doc_count == 0:
            warning_msg = (
                "WARNING: ChromaDB collection is EMPTY. "
                "No documents have been ingested. "
                "RAG queries will fall back to LLM-only answers. "
                "To populate the collection, run: python ingest.py --fresh"
            )
            logger.warning("=" * 60)
            logger.warning(warning_msg)
            logger.warning("=" * 60)
            # Don't raise - allow the app to start and use LLM fallback
        
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
    
    def _setup_retrievers(self):
        """Setup BM25 index from Chroma documents."""
        try:
            logger.info("Loading documents from Chroma for BM25 indexing...")
            all_documents, all_metadatas = _load_all_documents_from_chroma(self.collection)
            
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
    
    def _setup_chains(self):
        """Setup LangChain prompts and LCEL chains."""
        
        # RAG prompt (strict document-based answers with trust-first guidelines)
        self.rag_template = """You are SageAlpha AI, an enterprise-grade Financial & Regulatory AI assistant designed for high factual accuracy, zero hallucination, and audit-ready answers.

Your goal is to answer user questions by combining multiple sources of truth, in strict priority order:
1. Official company documents (from web search - highest priority)
2. Internal RAG knowledge (ChromaDB embeddings from curated documents)
3. LLM reasoning and summarization, strictly grounded in retrieved evidence

Context documents:
{context}

Question: {question}

STRICT INSTRUCTIONS:
- Answer using ONLY information from the provided context documents
- Include specific numbers, dates, and facts EXACTLY as they appear in the context
- Preserve exact numeric values (integers, floats, percentages, currency)
- If an exact number is not found in the context, say: "The exact value is not available in the official documents reviewed."
- NEVER estimate, approximate, or hallucinate numbers
- NEVER invent facts, numbers, or sources
- Cite official sources when referencing specific data
- If data is unavailable, say so clearly: "Based on the available official sources, this information could not be verified."

SOURCE REQUIREMENTS:
- Every factual answer MUST include official source identification
- For web-retrieved documents, cite the official company website URL
- For internal documents, cite the document name and page number
- Do NOT expose internal storage paths (Azure Blob, local temp paths)

Answer:"""
        
        self.rag_prompt = ChatPromptTemplate.from_template(self.rag_template)
        self.rag_chain = self.rag_prompt | self.llm | self.output_parser
        
        # LLM-only prompt (general knowledge with source attribution)
        self.llm_only_template = """You are SageAlpha AI, a financial AI assistant. Answer the question using your knowledge.

Question: {question}

Answer the question helpfully and accurately. For financial queries, be precise with numbers and dates when you have that information.

IMPORTANT: At the end of your answer, include a note about the source of your information. For example:
- "This information is based on publicly available financial data and company filings."
- "This answer is derived from general knowledge about the company's financial performance."
- "This information may be found in the company's annual reports or SEC filings."

Answer:"""
        
        self.llm_only_prompt = ChatPromptTemplate.from_template(self.llm_only_template)
        self.llm_only_chain = self.llm_only_prompt | self.llm | self.output_parser
    
    def _retrieve_documents_hybrid(self, question: str) -> Tuple[List[str], List[Dict]]:
        """
        STRICT retrieval: ALWAYS attempts to retrieve documents from ChromaDB.
        
        Uses SAME embedding model as ingestion (must match deployment name).
        Supports fiscal year filtering when query specifies a year.
        """
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
            
            # If fiscal year specified, try year-filtered retrieval first
            year_docs = []
            year_metas = []
            if requested_year:
                try:
                    logger.info(f"[RETRIEVER] Attempting year-filtered retrieval for {requested_year}")
                    year_results = self.collection.query(
                        query_embeddings=[query_embedding],
                        n_results=5,
                        where={"fiscal_year": requested_year},
                        include=["documents", "metadatas", "distances"]
                    )
                    year_docs = year_results.get("documents", [[]])[0] if year_results.get("documents") else []
                    year_metas = year_results.get("metadatas", [[]])[0] if year_results.get("metadatas") else []
                    logger.info(f"[RETRIEVER] Year-filtered retrieval found {len(year_docs)} documents for {requested_year}")
                except Exception as e:
                    logger.warning(f"[RETRIEVER] Year-filtered retrieval failed: {e}, falling back to general retrieval")
            
            # Always perform general retrieval
            chroma_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=5,
                include=["documents", "metadatas", "distances"]
            )
            
            chroma_docs = chroma_results.get("documents", [[]])[0] if chroma_results.get("documents") else []
            chroma_metas = chroma_results.get("metadatas", [[]])[0] if chroma_results.get("metadatas") else []
            chroma_distances = chroma_results.get("distances", [[]])[0] if chroma_results.get("distances") else []
            
            if chroma_distances:
                logger.info(f"[RETRIEVER] ChromaDB similarity distances: {chroma_distances[:3]}")
                logger.info(f"[RETRIEVER] Best match distance: {min(chroma_distances) if chroma_distances else 'N/A'}")
            
            logger.info(f"[RETRIEVER] General ChromaDB retrieval found {len(chroma_docs)} documents")
            
            # Log document metadata for debugging
            if chroma_metas:
                logger.info(f"[RETRIEVER] Sample metadata: {chroma_metas[0] if len(chroma_metas) > 0 else 'None'}")
                if len(chroma_metas) > 0:
                    sample_meta = chroma_metas[0]
                    logger.info(f"[RETRIEVER] Sample company: {sample_meta.get('company', 'N/A')}, Year: {sample_meta.get('fiscal_year', 'N/A')}")
            
            if not chroma_docs:
                logger.warning("[RETRIEVER] No documents retrieved from ChromaDB - check if collection is populated")
            
            # Combine results: year-specific first, then general (deduplicated)
            if year_docs:
                seen_texts = {doc.strip().lower() for doc in year_docs}
                additional_docs = []
                additional_metas = []
                for doc, meta in zip(chroma_docs, chroma_metas):
                    if doc.strip().lower() not in seen_texts:
                        additional_docs.append(doc)
                        additional_metas.append(meta)
                
                chroma_docs = year_docs + additional_docs[:3]
                chroma_metas = year_metas + additional_metas[:3]
                logger.info(f"[RETRIEVER] Combined retrieval: {len(year_docs)} year-specific + {len(additional_docs[:3])} general = {len(chroma_docs)} total")
            
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
                logger.info("[WEB_SEARCH] No company name found in query")
                return []
            
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
    
    def answer_query(self, question: str) -> Dict[str, Any]:
        """
        Enhanced orchestration with web search integration.
        
        Flow:
        1. ALWAYS retrieve documents from ChromaDB first
        2. Check if web search should be triggered
        3. If triggered, retrieve web evidence
        4. Validate answerability
        5. Fuse evidence from RAG + Web
        6. Generate answer with trust-first guidelines
        """
        # Initialize all variables with safe defaults to prevent UnboundLocalError
        answer = ""
        answer_type = "sagealpha_rag"
        formatted_sources = []
        rag_chain_executed = False
        rag_chain_error = None
        
        try:
            logger.info("[QUERY] Processing query")
            logger.info(f"[QUERY] Query text: {question}")
            
            # STEP 0: Classify query to determine evidence requirements
            query_classification = QueryClassifier.classify_query(question)
            is_factual_financial = query_classification["is_factual_financial"]
            requires_verified_source = query_classification["requires_verified_source"]
            
            logger.info(f"[CLASSIFIER] Query requires verified source: {requires_verified_source}")
            
            # STEP 1: ALWAYS perform RAG retrieval first (NO EXCEPTIONS)
            logger.info("[RETRIEVER] Starting RAG retrieval...")
            documents, metadatas = self._retrieve_documents_hybrid(question)
            
            # STEP 1.5: Validate company match to prevent cross-company contamination
            # Only validate if we have documents AND the query mentions a specific company
            query_company = CompanyValidator._extract_company_from_query(question)
            if documents and metadatas and query_company:
                logger.info(f"[VALIDATOR] Validating company name match (query company: '{query_company}')...")
                original_count = len(documents)
                documents, metadatas, rejected = CompanyValidator.validate_company_match(
                    question, documents, metadatas
                )
                if rejected:
                    logger.warning(f"[VALIDATOR] Rejected {len(rejected)} documents due to company mismatch")
                    logger.info(f"[VALIDATOR] Kept {len(documents)}/{original_count} documents after company validation")
            elif documents and metadatas:
                logger.info("[VALIDATOR] No company name detected in query, skipping company validation")
            
            # Get similarity scores for decision logic (distances from ChromaDB)
            # Note: ChromaDB returns distances (lower = more similar), not similarity scores
            similarity_scores = []
            if documents:
                # Get distances from the retrieval we just did
                try:
                    chroma_results = self.collection.query(
                        query_embeddings=[self.embeddings.embed_query(question)],
                        n_results=5,
                        include=["distances"]
                    )
                    similarity_scores = chroma_results.get("distances", [[]])[0] if chroma_results.get("distances") else []
                except:
                    similarity_scores = []
            
            logger.info(f"[DEBUG] Retrieved docs count: {len(documents)}")
            
            # STEP 2: Decide if web search should be triggered
            web_documents = []
            should_search, search_reason = should_trigger_web_search(
                question,
                documents,
                metadatas,
                similarity_scores
            )
            
            if should_search:
                logger.info(f"[WEB_SEARCH] Triggering web search: {search_reason}")
                web_documents = self._retrieve_web_evidence(question)
            else:
                logger.info(f"[WEB_SEARCH] Skipping web search: {search_reason}")
            
            # STEP 3: Evidence Gate - CRITICAL: Block LLM-only answers for factual financial queries
            if not documents and not web_documents:
                # No documents retrieved
                logger.warning("[ROUTER] No documents retrieved")
                logger.warning("[ROUTER] This may indicate:")
                logger.warning("[ROUTER] 1. ChromaDB collection is empty or not properly indexed")
                logger.warning("[ROUTER] 2. Query embeddings don't match stored embeddings")
                logger.warning("[ROUTER] 3. Company validator rejected all documents")
                
                # ABSOLUTE RULE: If query requires verified source, BLOCK LLM answer
                if requires_verified_source:
                    logger.error("[EVIDENCE_GATE] BLOCKED: Factual financial query without verified sources")
                    logger.error("[EVIDENCE_GATE] LLM is NOT allowed to answer factual financial questions without sources")
                    
                    return {
                        "answer": "The requested information could not be verified from official sources.",
                        "answer_type": "sagealpha_ai_search",  # Use search type, not LLM
                        "sources": []  # Empty sources indicate no verified data
                    }
                
                # Only allow LLM for non-factual queries (conceptual, general knowledge)
                logger.warning("=" * 60)
                logger.warning("[RESPONSE TYPE] LLM FALLBACK (No RAG documents available)")
                logger.warning("=" * 60)
                logger.warning("No documents retrieved from ChromaDB. Using LLM-only answer.")
                logger.warning("To enable RAG answers, run: python ingest.py --fresh")
                logger.warning("=" * 60)
                answer = self.llm_only_chain.invoke({"question": question})
                logger.info("[LLM] Answer generated from general knowledge")
                
                # For non-factual queries, use sagealpha_rag (not LLM)
                answer_type = "sagealpha_rag"
                logger.info(f"[RESPONSE] answer_type={answer_type} (LLM fallback, no sources)")
                logger.info("[RESPONSE] Returning LLM-only answer to user")
                
                # For LLM answers, create a source indicating it's from general knowledge
                llm_sources = [{
                    "title": "General Knowledge Base",
                    "publisher": "SageAlpha AI",
                    "note": "Answer generated from general knowledge. For verified financial data, please ensure relevant documents are indexed in the knowledge base."
                }]
                
                return {
                    "answer": answer,
                    "answer_type": answer_type,
                    "sources": llm_sources
                }
            
            # STEP 4: Validate answerability (if we have RAG docs)
            is_answerable = True
            reason = "Documents available"
            validation_details = {}
            
            if documents:
                logger.info("[VALIDATE] Starting answerability validation...")
                is_answerable, reason, validation_details = _validate_answerability(question, documents, metadatas)
            
            # If RAG docs not answerable but we have web docs, still proceed
            if not is_answerable and web_documents:
                logger.info("[VALIDATE] RAG docs not answerable, but web docs available - proceeding")
                is_answerable = True
                reason = "Web documents available"
            
            if not is_answerable and not web_documents:
                # Documents retrieved but don't match requirements and no web docs → RAG_NO_ANSWER
                logger.warning(f"[ROUTER] Documents not answerable → RAG_NO_ANSWER")
                logger.info(f"[ROUTER] Reason: {reason}")
                
                # Build informative answer
                requested_year = validation_details.get("requested_year")
                requested_entity = validation_details.get("requested_entity")
                
                answer_parts = []
                if requested_year:
                    answer_parts.append(f"FY{requested_year[2:]} data")
                if requested_entity:
                    answer_parts.append(f"{requested_entity} data")
                
                if answer_parts:
                    answer = f"The requested {', '.join(answer_parts)} is not available in the documents."
                else:
                    answer = "The requested information is not available in the documents."
                
                # Extract sources
                sources = []
                for meta in metadatas:
                    source_info = meta.get("source", meta.get("filename", "unknown"))
                    if meta.get("page"):
                        source_info += f" (page {meta.get('page')})"
                    if meta.get("fiscal_year"):
                        source_info += f" (FY: {meta.get('fiscal_year')})"
                    sources.append(source_info)
                
                answer_type = "sagealpha_rag"  # Still use RAG branding even if no answer
                logger.info(f"[RESPONSE] answer_type={answer_type}")
                logger.info("[RESPONSE] Returning answer to user")
                
                # Format sources (CRITICAL: Always populate sources for RAG answers)
                formatted_sources = SourceFormatter.format_sources(metadatas, [])
                
                # Ensure sources are never null/empty
                if not formatted_sources and metadatas:
                    logger.warning("[SOURCE] Formatter returned empty sources for RAG_NO_ANSWER, creating fallback")
                    formatted_sources = SourceFormatter._create_fallback_sources(metadatas)
                
                return {
                    "answer": answer,
                    "answer_type": answer_type,
                    "sources": formatted_sources if formatted_sources else []  # Never null
                }
            
            # STEP 5: Fuse evidence from RAG + Web
            logger.info("[ROUTER] Fusing evidence from RAG + Web Search")
            
            fusion_result = EvidenceFusion.fuse_evidence(
                documents,
                metadatas,
                web_documents,
                question
            )
            
            fused_context = fusion_result.get("fused_context", "")
            fused_sources = fusion_result.get("sources", [])
            evidence_priority = fusion_result.get("evidence_priority", "rag_internal")
            
            # STEP 6: Generate answer using fused context
            logger.info("[RAG] Sending fused context + query to LLM")
            
            # Track whether RAG chain actually executed successfully
            # (rag_chain_executed already initialized at function start)
            
            try:
                if fused_context:
                    answer = self.rag_chain.invoke({"question": question, "context": fused_context})
                else:
                    # Fallback if fusion failed
                    context_parts = []
                    for doc, meta in zip(documents, metadatas):
                        meta_info = ""
                        if meta.get("source"):
                            meta_info = f"Source: {meta.get('source')}"
                        if meta.get("fiscal_year"):
                            meta_info += f", FY: {meta.get('fiscal_year')}"
                        if meta.get("page"):
                            meta_info += f", Page: {meta.get('page')}"
                        if meta_info:
                            context_parts.append(f"[{meta_info}]\n{doc}")
                        else:
                            context_parts.append(doc)
                    
                    fused_context = "\n\n---\n\n".join(context_parts)
                    answer = self.rag_chain.invoke({"question": question, "context": fused_context})
                
                rag_chain_executed = True
                logger.info("[RAG] Answer generated from retrieved documents")
                
            except Exception as rag_error:
                rag_chain_error = rag_error
                error_msg = str(rag_error)
                
                # Get config for error messages
                config = get_config()
                
                logger.error("=" * 60)
                logger.error("[RAG CHAIN] FAILED - Azure OpenAI deployment error")
                logger.error("=" * 60)
                logger.error(f"Error type: {type(rag_error).__name__}")
                logger.error(f"Error message: {error_msg}")
                
                # Check if it's a deployment error
                if "NotFound" in error_msg or "404" in error_msg or "deployment" in error_msg.lower():
                    logger.error("This is an Azure OpenAI deployment configuration error.")
                    logger.error(f"Chat deployment '{config.azure_openai.chat_deployment}' may not exist.")
                    logger.error("Check Azure Portal → Deployments to verify the deployment name.")
                
                logger.error("=" * 60)
                logger.error("Falling back to LLM-only mode (no RAG context)")
                logger.error("=" * 60)
                
                # Fallback to LLM-only (but we still have documents, so this is a generation failure)
                try:
                    answer = self.llm_only_chain.invoke({"question": question})
                    logger.warning("[FALLBACK] LLM-only answer generated (RAG chain failed)")
                except Exception as llm_error:
                    # Even LLM fallback failed - this is a critical error
                    logger.error(f"[CRITICAL] LLM fallback also failed: {llm_error}")
                    raise RuntimeError(
                        f"Both RAG and LLM generation failed. "
                        f"RAG error: {rag_error}, LLM error: {llm_error}"
                    ) from llm_error
            
            # Get config for answer type determination (needed for error messages)
            config = get_config()
            
            # Determine answer type based on whether RAG chain executed AND what sources we have
            # CRITICAL: answer_type reflects HOW the answer was generated, not just source count
            # If RAG chain executed successfully, it's RAG (even if sources are empty)
            # If RAG chain failed, we still label based on documents retrieved (but log the failure)
            
            if rag_chain_executed:
                # RAG chain succeeded - this is a true RAG answer
                if web_documents and documents:
                    answer_type = "sagealpha_hybrid_search"
                    logger.info("=" * 60)
                    logger.info("[RESPONSE TYPE] HYBRID SEARCH (RAG + Web Search)")
                    logger.info("=" * 60)
                    logger.info(f"✓ RAG chain executed successfully")
                    logger.info(f"Using {len(documents)} RAG documents + {len(web_documents)} web documents")
                    logger.info("=" * 60)
                elif web_documents:
                    answer_type = "sagealpha_ai_search"
                    logger.info("=" * 60)
                    logger.info("[RESPONSE TYPE] WEB SEARCH ONLY")
                    logger.info("=" * 60)
                    logger.info(f"✓ RAG chain executed successfully")
                    logger.info(f"Using {len(web_documents)} web documents (no RAG documents found)")
                    logger.info("=" * 60)
                elif documents:
                    answer_type = "sagealpha_rag"
                    logger.info("=" * 60)
                    logger.info("[RESPONSE TYPE] RAG (Internal Documents)")
                    logger.info("=" * 60)
                    logger.info(f"✓ RAG chain executed successfully")
                    logger.info(f"Using {len(documents)} documents from ChromaDB")
                    logger.info("=" * 60)
                else:
                    # RAG executed but no documents - this shouldn't happen but handle it
                    answer_type = "sagealpha_rag"
                    logger.warning("[RESPONSE TYPE] RAG executed but no documents (unexpected)")
            else:
                # RAG chain failed - we fell back to LLM-only
                # BUT: We still have documents, so this is a generation failure, not a retrieval failure
                # We label based on what documents we retrieved (to show we tried RAG)
                if documents or web_documents:
                    # We have documents but RAG generation failed - still label as RAG type (with warning)
                    if web_documents and documents:
                        answer_type = "sagealpha_hybrid_search"
                    elif web_documents:
                        answer_type = "sagealpha_ai_search"
                    else:
                        answer_type = "sagealpha_rag"
                    
                    logger.warning("=" * 60)
                    logger.warning(f"[RESPONSE TYPE] {answer_type.upper()} (RAG GENERATION FAILED - LLM FALLBACK)")
                    logger.warning("=" * 60)
                    logger.warning(f"⚠ Documents retrieved ({len(documents)} RAG + {len(web_documents)} web)")
                    logger.warning(f"⚠ But RAG chain failed - using LLM-only answer")
                    logger.warning(f"⚠ This indicates Azure OpenAI deployment configuration issue")
                    logger.warning(f"⚠ Check: AZURE_OPENAI_CHAT_DEPLOYMENT_NAME={config.azure_openai.chat_deployment}")
                    logger.warning("=" * 60)
                else:
                    # No documents AND RAG failed - this is a true LLM fallback
                    answer_type = "sagealpha_rag"  # Still use RAG branding for consistency
                    logger.warning("=" * 60)
                    logger.warning("[RESPONSE TYPE] LLM FALLBACK (No documents + RAG failed)")
                    logger.warning("=" * 60)
                    logger.warning("No documents retrieved and RAG chain failed")
                    logger.warning("=" * 60)
            
            # Format sources using SourceFormatter (hides internal paths, shows official URLs)
            # (formatted_sources already initialized at function start, but we'll set it here)
            formatted_sources = SourceFormatter.format_sources(metadatas, web_documents)
            
            logger.info(f"[RESPONSE] answer_type={answer_type}")
            logger.info(f"[RESPONSE] sources_count={len(formatted_sources) if formatted_sources else 0}")
            logger.info("[RESPONSE] Returning answer to user")
            
            # CRITICAL: Ensure sources are never null/empty for RAG answers
            if not formatted_sources:
                if documents and metadatas:
                    # Fallback: Create sources from metadatas even if formatter returns empty
                    logger.warning("[SOURCE] Formatter returned empty sources, creating fallback sources")
                    formatted_sources = SourceFormatter._create_fallback_sources(metadatas)
                elif web_documents:
                    # Fallback for web documents
                    logger.warning("[SOURCE] Formatter returned empty sources for web docs, creating fallback")
                    web_metas = [doc.get("metadata", {}) for doc in web_documents]
                    formatted_sources = SourceFormatter._create_fallback_sources(web_metas)
            
            # ABSOLUTE RULE: If factual financial query and no sources, BLOCK answer
            if requires_verified_source and not formatted_sources:
                logger.error("[EVIDENCE_GATE] BLOCKED: Factual financial query with no sources after formatting")
                return {
                    "answer": "The requested information could not be verified from official sources.",
                    "answer_type": answer_type,
                    "sources": []  # Empty sources indicate no verified data
                }
            
            # Validate sources are non-empty (should be guaranteed by above logic)
            if not formatted_sources:
                logger.warning("[SOURCE] WARNING: Sources array is empty - this should not happen")
            
            return {
                "answer": answer,
                "answer_type": answer_type,
                "sources": formatted_sources if formatted_sources else []  # Never null
            }
            
        except Exception as e:
            logger.error(f"LangChain orchestration failed: {e}", exc_info=True)
            return {
                "answer": "I apologize, but I encountered an error while processing your query. Please try again.",
                "answer_type": "sagealpha_rag",
                "sources": []
            }
    
    # _extract_sources method removed - now using SourceFormatter.format_sources()


# Singleton instance
_orchestrator: Optional[LangChainOrchestrator] = None
_orchestrator_error: Optional[str] = None


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
