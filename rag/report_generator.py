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

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.settings import get_config
from vectorstore.chroma_client import get_collection
from rag.source_formatter import SourceFormatter

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
            azure_deployment=config.azure_openai.chat_deployment,
            api_key=config.azure_openai.api_key,
            api_version=config.azure_openai.api_version,
            temperature=0.3,  # Slightly creative for narrative
        )
        
        self.output_parser = StrOutputParser()
        
        # Get Chroma collections
        self.collection_names = ["dixon-technologies", "finance_documents"]
        self.collections = []
        self.chroma_available = False
        
        try:
            for name in self.collection_names:
                try:
                    col = get_collection(name=name, create_if_missing=False)
                    self.collections.append(col)
                    logger.info(f"[REPORT] Loaded collection: {name}")
                except Exception as e:
                    logger.warning(f"[REPORT] Collection '{name}' unavailable: {e}")
            
            if self.collections:
                self.chroma_available = True
                logger.info("[REPORT] Chroma collections connected successfully")
            else:
                 logger.warning("[REPORT] No collections available — skipping fact retrieval")
                
        except Exception:
            logger.warning("[REPORT] Chroma unavailable — disabling RAG")
            self.collections = []
            self.chroma_available = False
        
        # Setup report prompt
        self.report_template = """You are a Senior Equity Research Analyst.

Using the factual context provided below, generate a professional investment research report.

The report should be comprehensive, well-structured, and include:
- Executive Summary
- Company Overview
- Financial Analysis
- Investment Thesis
- Risks and Considerations
- Conclusion and Recommendation

FACTUAL CONTEXT:
{context}

USER REQUEST:
{query}

Generate a detailed, professional research report. Write in a clear, analytical style suitable for institutional investors.

Report:"""
        
        self.report_prompt = ChatPromptTemplate.from_template(self.report_template)
        self.report_chain = self.report_prompt | self.llm | self.output_parser
    
    def _retrieve_facts(self, query: str, n_results: int = 10) -> tuple[List[str], List[Dict]]:
        """
        Retrieve factual documents from Chroma.
        
        Args:
            query: User query
            n_results: Number of documents to retrieve
        
        Returns:
            (documents, metadatas)
        """
        if not getattr(self, "chroma_available", False):
            logger.warning("[REPORT] Chroma disabled — skipping fact retrieval")
            return [], []

        try:
            all_documents = []
            all_metadatas = []
            
            for col in self.collections:
                try:
                    results = col.query(
                        query_texts=[query],
                        n_results=n_results,
                        include=["documents", "metadatas", "distances"]
                    )
                    
                    docs = results.get("documents", [[]])[0] if results.get("documents") else []
                    metas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
                    
                    all_documents.extend(docs)
                    all_metadatas.extend(metas)
                except Exception as e:
                    logger.warning(f"[REPORT] Failed to retrieve from {col.name}: {e}")
            
            return all_documents, all_metadatas
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            return [], []
    
    def generate_report(self, query: str) -> Dict[str, Any]:
        """
        Generate a long-format report.
        
        Two-phase approach:
        1. Retrieve facts via RAG (if available)
        2. Always generate narrative via LLM
        
        This function NEVER blocks due to missing documents.
        """

        try:
            # Phase 1: Retrieve facts
            documents, metadatas = self._retrieve_facts(query, n_results=10)

            rag_used = len(documents) > 0

            if rag_used:
                logger.info("[REPORT] Generating report using retrieved RAG documents")
                fact_context = build_fact_context(documents, metadatas)
            else:
                logger.warning("[REPORT] No RAG documents found — switching to LLM-only mode")
                fact_context = (
                    "No verified internal documents available. "
                    "Generate report using general financial knowledge, "
                    "public market understanding, and structured investment analysis methodology."
                )

            # Phase 2: Generate report via LLM (always)
            report_text = self.report_chain.invoke({
                "query": query,
                "context": fact_context
            })

            # Format sources
            formatted_sources = []
            if rag_used:
                formatted_sources = SourceFormatter.format_sources(metadatas, [])
                if not formatted_sources and metadatas:
                    formatted_sources = SourceFormatter._create_fallback_sources(metadatas)

            return {
                "answer": report_text,
                "answer_type": "REPORT",
                "sources": formatted_sources,
                "rag_used": rag_used,
                "confidence_level": "HIGH" if rag_used else "LOW",
                "format": "long-format"
            }

        except Exception as e:
            logger.error(f"[REPORT] Critical failure — falling back to LLM-only: {e}", exc_info=True)

            try:
                fallback_text = self.report_chain.invoke({
                    "query": query,
                    "context": "Generate a professional investment research report using general financial knowledge."
                })

                return {
                    "answer": fallback_text,
                    "answer_type": "REPORT",
                    "sources": [],
                    "rag_used": False,
                    "confidence_level": "LOW",
                    "format": "llm_fallback"
                }

            except Exception as final_error:
                logger.error(f"[REPORT] Complete failure: {final_error}", exc_info=True)

                return {
                    "answer": "Report generation temporarily unavailable. Please try again.",
                    "answer_type": "REPORT",
                    "sources": [],
                    "rag_used": False,
                    "confidence_level": "LOW",
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
