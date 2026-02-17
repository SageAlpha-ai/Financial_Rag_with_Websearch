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
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=3,
                include=["documents", "metadatas", "distances"]
            )
            
            docs = results.get("documents", [[]])[0] if results.get("documents") else []
            metas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
            distances = results.get("distances", [[]])[0] if results.get("distances") else []
            
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

    def _plan_query(self, question: str, has_internal_knowledge: bool = False, internal_match_count: int = 0) -> Dict[str, Any]:
        """
        Evidence-Aware Planner.
        
        Decides routing strategy based on:
        1. Contextual signals (semantic match for RAG)
        2. Query intent (real-time vs historical)
        """
        default_plan = {"mode": "rag", "reason": "Default fallback"}
        
        try:
            if not self.planner_llm:
                 return default_plan

            planner_prompt = f"""You are an intelligent AI routing engine.

You must decide how to answer a query.

Signals provided:

- has_internal_knowledge: {has_internal_knowledge}
- internal_match_count: {internal_match_count}
- Query: "{question}"

Available modes:

- "rag"     → Use internal knowledge only.
- "web"     → Use web search only.
- "hybrid"  → Use both internal knowledge and web.
- "llm"     → Use model knowledge only.

Rules:

If strong internal semantic match exists and query is historical → prefer "rag".

If query requires current, live, today, latest, realtime info → prefer "web".

If both internal historical data AND latest update required → choose "hybrid".

If no retrieval required → choose "llm".

Return ONLY JSON:

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
            if has_internal_knowledge:
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
            
            # STEP 1: DETECT SUPPORTED COMPANY
            matched_company = None
            if config.supported_companies:
                for company in config.supported_companies:
                    if company in question_lower:
                        matched_company = company
                        break
            
            # Initialize evidence buckets
            rag_docs = []
            rag_metas = []
            web_docs = []
            distances = []
            
            # STEP 2: IF SUPPORTED COMPANY -> TRY SEMANTIC RAG
            if matched_company:
                logger.info(f"[ROUTER] Supported company detected: {matched_company}")
                logger.info("[ROUTER] Running semantic similarity search in ChromaDB")
                
                # Perform retrieval (we use existing method which handles embeddings + year filtering)
                # Note: To get scores, we need to query collection directly or modify retriever.
                # For simplicity and robustness, we use the raw collection query here for routing decision.
                
                try:
                    query_embedding = self.embeddings.embed_query(question)
                    results = self.collection.query(
                        query_embeddings=[query_embedding],
                        n_results=5,
                        include=["documents", "metadatas", "distances"]
                    )
                    
                    raw_docs = results.get("documents", [[]])[0] if results.get("documents") else []
                    raw_metas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
                    raw_distances = results.get("distances", [[]])[0] if results.get("distances") else []
                    
                    # Store for use
                    rag_docs = raw_docs
                    rag_metas = raw_metas
                    distances = raw_distances
                    
                    # Calculate match strength
                    # Chroma returns L2/Cosine distance. Lower is better.
                    # Assumption: 0.0 is exact match.
                    # User threshold: Similarity >= 0.75.
                    # If using Cosine Distance (range 0-1 usually, max 2): Similarity ~= 1 - Distance.
                    # So 1 - Distance >= 0.75 => Distance <= 0.25.
                    # We'll use a configurable threshold logic.
                    
                    is_strong_match = False
                    if distances:
                        min_distance = min(distances)
                        logger.info(f"[ROUTER] Best semantic distance: {min_distance}")
                        
                        # Threshold: 0.35 is a reasonable strict cutoff for ada-002 / text-embedding-3-small
                        # Adjusting to be roughly equivalent to 0.75 similarity
                        if min_distance <= 0.4:
                            is_strong_match = True
                            logger.info(f"[ROUTER] Strong semantic match (distance {min_distance} <= 0.4)")
                        else:
                            logger.info(f"[ROUTER] Weak semantic match (distance {min_distance} > 0.4)")
                    
                except Exception as e:
                    logger.error(f"[ROUTER] Semantic search failed: {e}")
                    is_strong_match = False
                
                if is_strong_match:
                    # RAG ONLY PATH
                    mode = "rag"
                    # rag_docs and rag_metas are already populated
                else:
                    # WEAK/NO MATCH -> HYBRID PATH
                    mode = "hybrid"
                    logger.info("[ROUTER] Switching to HYBRID (Web + LLM)")
                    web_docs = self._retrieve_web_evidence(question)
            
            # STEP 3: UNSUPPORTED COMPANY -> WEB + LLM
            else:
                mode = "web"
                logger.info("[ROUTER] Unsupported/General query -> using WEB + LLM")
                web_docs = self._retrieve_web_evidence(question)

                # Fallback to LLM only if web fails
                if not web_docs:
                    mode = "llm"
            
            logger.info(f"[ROUTER] Final Mode: {mode.upper()}")
            
            # STEP 4: EXECUTE & SYNTHESIZE
            
            if mode == "llm":
                logger.info("[ROUTER] LLM-only execution")
                chain = get_llm_only_chain()
                answer = chain.invoke({"question": question})
                return {
                    "answer": answer,
                    "answer_type": "sagealpha_llm",
                    "sources": []
                }
            
            # Determine answer type string
            if rag_docs and web_docs:
                answer_type = "sagealpha_hybrid_search"
            elif rag_docs:
                answer_type = "sagealpha_rag"
            elif web_docs:
                answer_type = "sagealpha_ai_search"
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
            system_prompt = """You are a financial AI assistant.

Use the following context sources to answer the user question.

CONTEXT:
{context}

Guidelines:
1. Synthesize information from the provided context.
2. If web data is more recent (e.g., stock price), prioritize it.
3. If internal documents provide specific details, prioritize them.
4. If no context is provided, answer using your general knowledge but mention that no specific documents were found.

Question:
{question}

Answer:"""

            generation_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("user", "{question}")
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
                "sources": formatted_sources if formatted_sources else []
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
