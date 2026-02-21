"""
Report Generation Module

Handles formatting of long-form reports using context provided by the orchestrator.
No longer performs direct retrieval.
"""

import logging
from typing import Dict, Any, List, Optional
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.settings import get_config
from rag.source_formatter import SourceFormatter

logger = logging.getLogger(__name__)

class ReportGenerator:
    """
    Generates detailed reports based on provided context.
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
            temperature=0.3,
        )
        
        self.output_parser = StrOutputParser()
        
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
    
    def generate_report(self, query: str, context: str, sources: List[Any] = None) -> Dict[str, Any]:
        """
        Generate a long-format report using provided context.
        """
        try:
            logger.info("[REPORT] Generating report from provided context")
            
            # Generate report via LLM
            report_text = self.report_chain.invoke({
                "query": query,
                "context": context
            })

            return {
                "answer": report_text,
                "answer_type": "REPORT",
                "sources": sources if sources else [],
                "format": "long-format"
            }

        except Exception as e:
            logger.error(f"[REPORT] Generation failed: {e}", exc_info=True)
            return {
                "answer": "Report generation failed. Please try again.",
                "answer_type": "REPORT",
                "sources": [],
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

def generate_report(query: str, context: str, sources: List[Any] = None) -> Dict[str, Any]:
    """
    Generate a long-format report using context.
    """
    generator = get_report_generator()
    return generator.generate_report(query, context, sources)
