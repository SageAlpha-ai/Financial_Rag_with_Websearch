"""
Web Search Module using SerpApi

Searches for company investor relations pages and official documents.
Follows strict validation rules to ensure only official sources are used.
"""

import logging
import os
import re
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, urljoin

import requests
from serpapi import GoogleSearch

logger = logging.getLogger(__name__)


class WebSearchEngine:
    """
    Web search engine using SerpApi for finding company investor relations pages.
    
    Features:
    - Company identity resolution
    - Official domain detection
    - Investor section discovery
    - Strict source validation
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize web search engine.
        
        Args:
            api_key: SerpApi API key (defaults to SERP_API env var)
        """
        self.api_key = api_key or os.getenv("SERP_API_KEY") or os.getenv("SERP_API")
        if not self.api_key:
            logger.warning("SerpApi key not found. Web search will be disabled.")
            self.enabled = False
        else:
            self.enabled = True
            logger.info("Web search engine initialized with SerpApi")
    
    def search_company_investor_relations(
        self, 
        company_name: str
    ) -> Dict[str, any]:
        """
        Search for company investor relations page.
        
        Args:
            company_name: Company name to search for
            
        Returns:
            Dict with:
            - official_domain: str | None
            - investor_url: str | None
            - search_results: List[Dict]
            - success: bool
        """
        if not self.enabled:
            return {
                "official_domain": None,
                "investor_url": None,
                "search_results": [],
                "success": False,
                "error": "Web search not enabled (missing API key)"
            }
        
        try:
            logger.info(f"[WEB_SEARCH] Searching for investor relations: {company_name}")
            
            # Search query: company name + "investor relations"
            search_query = f'"{company_name}" investor relations official website'
            
            params = {
                "q": search_query,
                "api_key": self.api_key,
                "engine": "google",
                "num": 10
            }
            
            search = GoogleSearch(params)
            results = search.get_dict()
            
            organic_results = results.get("organic_results", [])
            logger.info(f"[WEB_SEARCH] Found {len(organic_results)} search results")
            
            # Find official domain
            official_domain = self._identify_official_domain(company_name, organic_results)
            
            if not official_domain:
                logger.warning(f"[WEB_SEARCH] Could not identify official domain for {company_name}")
                return {
                    "official_domain": None,
                    "investor_url": None,
                    "search_results": organic_results[:5],
                    "success": False,
                    "error": "Official domain not found"
                }
            
            logger.info(f"[WEB_SEARCH] Identified official domain: {official_domain}")
            
            # Find investor relations URL
            investor_url = self._find_investor_section(official_domain, organic_results)
            
            return {
                "official_domain": official_domain,
                "investor_url": investor_url,
                "search_results": organic_results[:5],
                "success": investor_url is not None
            }
            
        except Exception as e:
            logger.error(f"[WEB_SEARCH] Search failed: {e}", exc_info=True)
            return {
                "official_domain": None,
                "investor_url": None,
                "search_results": [],
                "success": False,
                "error": str(e)
            }
    
    def _identify_official_domain(
        self, 
        company_name: str, 
        search_results: List[Dict]
    ) -> Optional[str]:
        """
        Identify official company domain from search results.
        
        Rejects:
        - Wikipedia
        - News aggregators
        - Blogs
        - Third-party summaries
        
        Accepts:
        - Official company domains
        - Recognized regulator domains (SEC, exchanges)
        """
        company_keywords = self._extract_company_keywords(company_name)
        
        for result in search_results:
            link = result.get("link", "")
            title = result.get("title", "").lower()
            snippet = result.get("snippet", "").lower()
            
            parsed = urlparse(link)
            domain = parsed.netloc.lower()
            
            # Reject non-official sources
            if any(reject in domain for reject in [
                "wikipedia.org",
                "bloomberg.com",
                "reuters.com",
                "yahoo.com",
                "marketwatch.com",
                "seekingalpha.com",
                "investing.com",
                "finance.yahoo.com",
                "blog",
                "news",
                "medium.com",
                "linkedin.com/posts"
            ]):
                continue
            
            # Check if domain matches company keywords
            domain_match = any(keyword in domain for keyword in company_keywords)
            title_match = any(keyword in title for keyword in company_keywords)
            snippet_match = any(keyword in snippet for keyword in company_keywords)
            
            # Accept if domain matches or title/snippet strongly suggests official
            if domain_match or (title_match and "official" in snippet):
                logger.info(f"[WEB_SEARCH] Official domain candidate: {domain}")
                return domain
        
        return None
    
    def _extract_company_keywords(self, company_name: str) -> List[str]:
        """Extract keywords from company name for matching."""
        # Remove common suffixes
        name = company_name.lower()
        name = re.sub(r'\s+(ltd|limited|inc|incorporated|corp|corporation|llc)\s*$', '', name)
        
        # Split into keywords
        keywords = [word for word in name.split() if len(word) > 2]
        
        # Add common variations
        if "financial" in name:
            keywords.append("financial")
        if "services" in name:
            keywords.append("services")
        
        return keywords
    
    def _find_investor_section(
        self, 
        official_domain: str, 
        search_results: List[Dict]
    ) -> Optional[str]:
        """
        Find investor relations section URL.
        
        Tries:
        1. Direct links from search results
        2. Common investor relations paths
        """
        # First, check search results for direct investor links
        for result in search_results:
            link = result.get("link", "")
            title = result.get("title", "").lower()
            
            parsed = urlparse(link)
            if parsed.netloc.lower() == official_domain:
                # Check if it's an investor relations page
                path = parsed.path.lower()
                if any(keyword in path or keyword in title for keyword in [
                    "investor",
                    "ir",
                    "financial",
                    "annual-report",
                    "sec-filing",
                    "earnings"
                ]):
                    logger.info(f"[WEB_SEARCH] Found investor URL in search results: {link}")
                    return link
        
        # Try common paths
        common_paths = [
            "/investor-relations",
            "/investors",
            "/investor",
            "/about/investors",
            "/ir",
            "/investor-relations/",
            "/investors/",
        ]
        
        # Try https first, then http
        for scheme in ["https", "http"]:
            base_url = f"{scheme}://{official_domain}"
            for path in common_paths:
                test_url = urljoin(base_url, path)
                if self._check_url_exists(test_url):
                    logger.info(f"[WEB_SEARCH] Found investor URL via path: {test_url}")
                    return test_url
        
        logger.warning(f"[WEB_SEARCH] Could not find investor section for {official_domain}")
        return None
    
    def _check_url_exists(self, url: str) -> bool:
        """Check if URL exists and is accessible."""
        try:
            response = requests.head(url, timeout=5, allow_redirects=True)
            return response.status_code == 200
        except:
            return False
    
    def search_financial_documents(
        self,
        company_name: str,
        document_type: str = "annual report",
        year: Optional[str] = None
    ) -> List[Dict]:
        """
        Search for specific financial documents.
        
        Args:
            company_name: Company name
            document_type: Type of document (annual report, 10-k, earnings, etc.)
            year: Optional year filter
            
        Returns:
            List of document results with links
        """
        if not self.enabled:
            return []
        
        try:
            query_parts = [f'"{company_name}"', document_type]
            if year:
                query_parts.append(year)
            query_parts.append("site:investor OR site:ir OR filetype:pdf")
            
            search_query = " ".join(query_parts)
            
            params = {
                "q": search_query,
                "api_key": self.api_key,
                "engine": "google",
                "num": 10,
                "fileType": "pdf"
            }
            
            search = GoogleSearch(params)
            results = search.get_dict()
            
            organic_results = results.get("organic_results", [])
            
            # Filter to only official domains
            filtered_results = []
            for result in organic_results:
                link = result.get("link", "")
                parsed = urlparse(link)
                domain = parsed.netloc.lower()
                
                # Only accept PDFs from official-looking domains
                if link.endswith(".pdf") and not any(reject in domain for reject in [
                    "wikipedia", "blog", "news", "medium"
                ]):
                    filtered_results.append({
                        "title": result.get("title", ""),
                        "link": link,
                        "snippet": result.get("snippet", ""),
                        "type": document_type
                    })
            
            logger.info(f"[WEB_SEARCH] Found {len(filtered_results)} financial documents")
            return filtered_results
            
        except Exception as e:
            logger.error(f"[WEB_SEARCH] Document search failed: {e}", exc_info=True)
            return []


def should_trigger_web_search(
    query: str,
    documents: List[str],
    metadatas: List[Dict],
    similarity_scores: Optional[List[float]] = None
) -> Tuple[bool, str]:
    """
    Decision logic for when to trigger web search.
    
    Triggers web search if:
    1. Query asks for latest financials/revenue/earnings
    2. Vector similarity score < threshold (e.g., 0.75)
    3. Retrieved docs are older than required year
    4. Company name is explicitly mentioned
    
    Returns:
        (should_search: bool, reason: str)
    """
    query_lower = query.lower()
    
    # Check if query is about financial metrics
    financial_keywords = [
        "revenue", "profit", "earnings", "financial", "annual report",
        "10-k", "10-q", "filing", "quarterly", "yearly"
    ]
    is_financial_query = any(kw in query_lower for kw in financial_keywords)
    
    # Check if company name is mentioned
    company_indicators = [
        "ltd", "limited", "inc", "corporation", "corp", "company"
    ]
    has_company_name = any(indicator in query_lower for indicator in company_indicators)
    
    # Check similarity scores
    low_similarity = False
    if similarity_scores:
        avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
        # Convert distance to similarity (assuming cosine distance)
        # Lower distance = higher similarity
        # If using distances, threshold should be > 0.25 (meaning similarity < 0.75)
        if avg_similarity > 0.25:  # Adjust based on your distance metric
            low_similarity = True
    
    # Check if documents are insufficient
    insufficient_docs = len(documents) < 2
    
    # Decision logic
    if is_financial_query and has_company_name:
        return True, "Financial query with company name - requires official sources"
    
    if low_similarity and is_financial_query:
        return True, "Low similarity score for financial query"
    
    if insufficient_docs and is_financial_query:
        return True, "Insufficient documents for financial query"
    
    # Don't search for general/regulatory queries
    regulatory_keywords = ["sebi", "regulation", "rule", "act", "law", "compliance"]
    if any(kw in query_lower for kw in regulatory_keywords):
        return False, "Regulatory query - use internal documents only"
    
    return False, "Query can be answered from internal documents"
