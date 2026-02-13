"""
Report Generation Module

Two-phase pipeline for generating long-format reports and PDFs:
1. RAG: Extract factual context from documents
2. LLM: Generate structured narrative reports

Separates retrieval (facts) from generation (narrative).
"""

import logging
import json
from typing import Dict, Any, List, Optional

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.settings import get_config
from vectorstore.chroma_client import get_collection
from rag.source_formatter import SourceFormatter
from rag.metrics import MetricsRecorder

logger = logging.getLogger(__name__)


def is_report_request(text: str) -> bool:
    """
    Detect if user is requesting a report (long-format) vs Q&A.
    
    Args:
        text: User input text
    
    Returns:
        True if report intent detected, False for Q&A
    """
    text_lower = text.lower()
    
    report_keywords = [
        "report", "investment", "analysis", "research",
        "valuation", "recommendation", "pdf",
        "equity research", "financial analysis",
        "long format", "detailed analysis",
        "generate report", "create report"
    ]
    
    return any(keyword in text_lower for keyword in report_keywords)


def build_fact_context(documents: List[str], metadatas: List[Dict]) -> str:
    """
    Build clean fact context from retrieved chunks.
    
    Summarizes retrieved documents into a structured fact list
    suitable for report generation (not raw chunks).
    
    Args:
        documents: Retrieved document chunks
        metadatas: Document metadata
    
    Returns:
        Clean fact context string
    """
    if not documents:
        return "No relevant documents found."
    
    fact_parts = []
    for i, (doc, meta) in enumerate(zip(documents[:10], metadatas[:10]), 1):
        meta_info = []
        if meta.get("source"):
            meta_info.append(f"Source: {meta.get('source')}")
        if meta.get("fiscal_year"):
            meta_info.append(f"FY: {meta.get('fiscal_year')}")
        if meta.get("company"):
            meta_info.append(f"Company: {meta.get('company')}")
        
        meta_str = f" [{', '.join(meta_info)}]" if meta_info else ""
        fact_parts.append(f"{i}. {doc.strip()}{meta_str}")
    
    return "\n\n".join(fact_parts)


class ReportGenerator:
    """
    Report generation using two-phase approach:
    1. RAG: Retrieve facts
    2. LLM: Generate narrative report
    """
    
    def __init__(self):
        """Initialize report generator."""
        config = get_config()
        
        # Azure OpenAI LLM (for report generation)
        self.llm = AzureChatOpenAI(
            azure_endpoint=config.azure_openai.endpoint,
            azure_deployment=config.azure_openai.large_chat_deployment,
            api_key=config.azure_openai.api_key,
            api_version=config.azure_openai.api_version,
            temperature=0.3,  # Slightly creative for narrative
        )
        
        self.output_parser = StrOutputParser()
        self.collection = get_collection(create_if_missing=False)
        
        # Embedding client — same deployment as ingestion and the orchestrator
        # to ensure vector-space consistency.
        embedding_kwargs = {
            "azure_endpoint": config.azure_openai.endpoint,
            "azure_deployment": config.azure_openai.embeddings_deployment,
            "api_key": config.azure_openai.api_key,
            "api_version": config.azure_openai.api_version,
        }
        if "text-embedding-ada-002" in config.azure_openai.embeddings_deployment.lower():
            embedding_kwargs["model"] = "text-embedding-ada-002"
        self.embeddings = AzureOpenAIEmbeddings(**embedding_kwargs)
        
        # Setup report prompt
        self.report_template = """You are a Senior Equity Research Analyst.

Using ONLY the factual context provided below, generate a professional investment research report.

The report MUST be grounded entirely in the provided documents.
Do NOT invent, estimate, or hallucinate any financial figures, metrics, or data points.
If a standard report section (e.g. Financial Analysis, Valuation) cannot be supported
by the provided context, omit that section or explicitly state that data is unavailable.

Include only sections for which factual evidence exists:
- Executive Summary
- Company Overview
- Financial Analysis (only if numeric data is present)
- Investment Thesis (only if supported by evidence)
- Risks and Considerations
- Conclusion and Recommendation

FACTUAL CONTEXT:
{context}

USER REQUEST:
{query}

Generate a detailed, professional research report using only the facts above. Write in a clear, analytical style suitable for institutional investors. If required financial data is not present in the context, state that the information is not available rather than estimating.

Report:"""
        
        self.report_prompt = ChatPromptTemplate.from_template(self.report_template)
        self.report_chain = self.report_prompt | self.llm | self.output_parser
    
    def _retrieve_facts(self, query: str, n_results: int = 10) -> tuple[List[str], List[Dict]]:
        """
        Retrieve factual documents from Chroma using embedding-based search.

        Uses the same Azure OpenAI embedding model as ingestion and the
        orchestrator to ensure vector-space consistency.
        
        Args:
            query: User query
            n_results: Number of documents to retrieve
        
        Returns:
            (documents, metadatas)
        """
        try:
            query_embedding = self.embeddings.embed_query(query)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            documents = results.get("documents", [[]])[0] if results.get("documents") else []
            metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
            
            logger.info(
                "[REPORT] Embedding-based retrieval returned %d documents",
                len(documents),
            )
            return documents, metadatas
        except Exception as e:
            logger.error("[REPORT] Document retrieval failed: %s", e)
            return [], []
    
    def generate_report(self, query: str) -> Dict[str, Any]:
        """
        Generate a long-format report.
        
        Two-phase approach:
        1. Retrieve facts via RAG
        2. Generate narrative via LLM
        
        Args:
            query: User query/request
        
        Returns:
            Dict with report text and metadata
        """
        try:
            # Phase 1: Retrieve facts
            documents, metadatas = self._retrieve_facts(query, n_results=10)

            # ------------------------------------------------------------ #
            # Evidence gate: report generation is higher-risk than Q&A.     #
            # If no verified documents were retrieved, we MUST NOT invoke   #
            # the LLM report chain — doing so would produce an entirely     #
            # hallucinated financial report.                                 #
            # ------------------------------------------------------------ #
            if not documents:
                logger.warning(
                    "[REPORT] No verified documents retrieved — "
                    "blocking report generation"
                )
                MetricsRecorder.record_insufficient_evidence(
                    query, reason="no_documents_for_report",
                )
                return {
                    "answer": (
                        "Insufficient verified documents to generate a "
                        "financial report. Please ensure relevant financial "
                        "documents have been ingested, or try a more specific "
                        "query."
                    ),
                    "answer_type": "sagealpha_rag",
                    "sources": [],
                    "format": "insufficient_evidence"
                }

            fact_context = build_fact_context(documents, metadatas)
            
            # Phase 2: Generate narrative report (only with verified evidence)
            report_text = self.report_chain.invoke({
                "query": query,
                "context": fact_context
            })
            
            # Format sources using SourceFormatter
            formatted_sources = SourceFormatter.format_sources(metadatas, [])
            
            # Ensure sources are never null
            if not formatted_sources and metadatas:
                formatted_sources = SourceFormatter._create_fallback_sources(metadatas)
            
            return {
                "answer": report_text,
                "answer_type": "sagealpha_rag",  # Reports use RAG branding
                "sources": formatted_sources if formatted_sources else [],  # Never null
                "format": "long-format"
            }
        except Exception as e:
            logger.error(f"Report generation failed: {e}", exc_info=True)
            # Fallback to shorter response
            return {
                "answer": f"I apologize, but I encountered an error while generating the report. Please try rephrasing your request.",
                "answer_type": "sagealpha_rag",  # Reports use RAG branding
                "sources": [],  # Never null
                "format": "error"
            }


# Singleton instance
_report_generator: Optional[ReportGenerator] = None


def get_report_generator() -> ReportGenerator:
    """Get singleton report generator instance."""
    global _report_generator
    if _report_generator is None:
        _report_generator = ReportGenerator()
    return _report_generator


def generate_report(query: str) -> Dict[str, Any]:
    """
    Generate a long-format report.
    
    Uses two-phase pipeline: RAG for facts, LLM for narrative.
    """
    generator = get_report_generator()
    return generator.generate_report(query)
