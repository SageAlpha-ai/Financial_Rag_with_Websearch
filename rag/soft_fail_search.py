"""
Soft-Fail Web Search Integration.

Wraps the WebSearchEngine with resilience and gracefully handles 
API failures, timeouts, and low-quality results by falling back 
to the next layer (LLM Knowledge).
"""

import logging
from typing import List, Dict, Optional
from .web_search import WebSearchEngine

logger = logging.getLogger(__name__)

class SoftFailWebSearch:
    """Resilient wrapper for web search operations."""

    def __init__(self, engine: WebSearchEngine):
        self.engine = engine

    def search_with_fallback(
        self, 
        query: str, 
        company_name: Optional[str] = None
    ) -> List[Dict]:
        """
        Executes web search with soft-fail logic.
        
        Failures (API errors, empty results) do NOT raise exceptions but
        return an empty list, allowing the orchestrator to proceed to
        LLM Knowledge fallback.
        """
        if not self.engine.enabled:
            logger.info("[SOFT_FAIL] Web search is disabled. Skipping.")
            return []

        try:
            logger.info(f"[SOFT_FAIL] Attempting web search for: {query}")
            
            # 1. Try company-specific search if company is known
            if company_name:
                ir_results = self.engine.search_company_investor_relations(company_name)
                if ir_results.get("success") and ir_results.get("investor_url"):
                    # Here we could trigger scraping, but for 'soft-fail' we return what's found
                    # In a real integration, this would call investor_scraper
                    pass

            # 2. Execute phased search
            results = self.engine._phased_search(query)
            
            if not results:
                logger.warning(f"[SOFT_FAIL] No results found for query: {query}")
                return []

            # 3. Format results as 'web' documents
            web_docs = []
            for res in results[:5]:
                web_docs.append({
                    "text": res.get("snippet", ""),
                    "metadata": {
                        "title": res.get("title", ""),
                        "url": res.get("link", ""),
                        "source": "web_search",
                        "type": "web"
                    }
                })
            
            return web_docs

        except Exception as e:
            # SOFT FAIL: Log the error but don't crash
            logger.error(f"[SOFT_FAIL] Web search failed critically: {e}")
            return []

    @staticmethod
    def get_disclosure_note(has_web_results: bool) -> Optional[str]:
        """Returns the appropriate disclosure based on result availability."""
        if has_web_results:
            return "Answer supplemented with data from public web sources. Accuracy may vary."
        return None
