"""
Web Search Module using SerpApi

Searches for company investor relations pages and official documents.
Follows strict validation rules to ensure only official sources are used.

Source‑priority configuration (PRIORITY_WEB_DOMAINS, ALLOWED_FALLBACK_DOMAINS,
BLOCKED_DOMAINS) is defined centrally in config/settings.py.
"""

import logging
import os
import re
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, urljoin

import requests
from serpapi import GoogleSearch

from config import PRIORITY_WEB_DOMAINS, ALLOWED_FALLBACK_DOMAINS, BLOCKED_DOMAINS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain helper utilities (module‑level, stateless)
# ---------------------------------------------------------------------------

def _extract_domain(url: str) -> str:
    """Return the lowercase netloc (domain) of *url*."""
    return urlparse(url).netloc.lower()


def _matches_domain_list(domain: str, domain_list: List[str]) -> bool:
    """
    Return True if *domain* matches any entry in *domain_list*.

    An entry is matched when:
      • the domain equals the entry, OR
      • the domain ends with "." + entry  (sub‑domain match), OR
      • the entry starts with "." and the domain ends with that suffix.
    """
    for entry in domain_list:
        if entry.startswith("."):
            # Suffix match (e.g. ".gov.in")
            if domain.endswith(entry) or domain.endswith(entry.lstrip(".")):
                return True
        else:
            if domain == entry or domain.endswith("." + entry):
                return True
    return False


def _is_blocked(domain: str) -> bool:
    """Return True if *domain* belongs to a blocked source."""
    return _matches_domain_list(domain, BLOCKED_DOMAINS)


def _is_priority_domain(domain: str) -> bool:
    """Return True if *domain* belongs to a priority source."""
    return _matches_domain_list(domain, PRIORITY_WEB_DOMAINS)


def _is_fallback_domain(domain: str) -> bool:
    """Return True if *domain* matches an allowed fallback suffix."""
    return _matches_domain_list(domain, ALLOWED_FALLBACK_DOMAINS)


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
        self.api_key = api_key or os.getenv("SERP_API_KEY")
        if not self.api_key:
            logger.warning("SerpApi key not found. Web search will be disabled.")
            self.enabled = False
        else:
            self.enabled = True
            logger.info("Web search engine initialized with SerpApi")

    # ------------------------------------------------------------------
    # Phased search & result filtering
    # ------------------------------------------------------------------

    @staticmethod
    def _build_site_restriction(domains: List[str]) -> str:
        """Build a Google ``site:`` restriction string for *domains*.

        Entries that start with ``"."`` (suffix matches such as ``".gov.in"``)
        have the leading dot stripped so the ``site:`` operator covers all
        sub‑domains (e.g. ``site:gov.in``).
        """
        parts = []
        for d in domains:
            clean = d.lstrip(".")
            parts.append(f"site:{clean}")
        return " OR ".join(parts)

    def _execute_serpapi_query(self, query: str, num: int = 10) -> List[Dict]:
        """Execute a single SerpApi Google query and return organic results."""
        params = {
            "q": query,
            "api_key": self.api_key,
            "engine": "google",
            "num": num,
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        return results.get("organic_results", [])

    def _filter_results(self, results: List[Dict], *, phase: str = "") -> List[Dict]:
        """Filter *results* through blocked / priority / fallback domain rules.

        Returns only results whose domains are:
          1. NOT in BLOCKED_DOMAINS, **and**
          2. in PRIORITY_WEB_DOMAINS or matching ALLOWED_FALLBACK_DOMAINS.
        """
        filtered: List[Dict] = []
        for result in results:
            link = result.get("link", "")
            domain = _extract_domain(link)

            if _is_blocked(domain):
                logger.debug(
                    "[WEB_SEARCH][%s] Rejected blocked domain: %s (%s)",
                    phase, domain, link,
                )
                continue

            if not (_is_priority_domain(domain) or _is_fallback_domain(domain)):
                logger.debug(
                    "[WEB_SEARCH][%s] Rejected non‑priority/non‑fallback domain: %s (%s)",
                    phase, domain, link,
                )
                continue

            filtered.append(result)
        return filtered

    def _phased_search(self, base_query: str, num: int = 10) -> List[Dict]:
        """Execute a two‑phase search respecting source‑priority config.

        Phase 1 – search restricted to PRIORITY_WEB_DOMAINS.
        Phase 2 – (only when Phase 1 yields 0 results) search restricted to
                  ALLOWED_FALLBACK_DOMAINS.

        All results are additionally filtered through ``_filter_results``.
        """
        # --- Phase 1: priority domains ---
        site_clause = self._build_site_restriction(PRIORITY_WEB_DOMAINS)
        phase1_query = f"{base_query} ({site_clause})"
        raw_results = self._execute_serpapi_query(phase1_query, num=num)
        filtered = self._filter_results(raw_results, phase="Phase1")

        if filtered:
            logger.info(
                "[WEB_SEARCH] Phase 1 returned %d result(s) from priority domains.",
                len(filtered),
            )
            return filtered

        # --- Phase 2: fallback domains ---
        logger.debug(
            "[WEB_SEARCH] Phase 1 returned 0 results. Falling back to Phase 2 "
            "(ALLOWED_FALLBACK_DOMAINS: %s).",
            ALLOWED_FALLBACK_DOMAINS,
        )
        site_clause = self._build_site_restriction(ALLOWED_FALLBACK_DOMAINS)
        phase2_query = f"{base_query} ({site_clause})"
        raw_results = self._execute_serpapi_query(phase2_query, num=num)
        filtered = self._filter_results(raw_results, phase="Phase2")

        if filtered:
            logger.info(
                "[WEB_SEARCH] Phase 2 returned %d result(s) from fallback domains.",
                len(filtered),
            )
            return filtered

        logger.info("[WEB_SEARCH] Both phases returned 0 results.")
        return []

    # ------------------------------------------------------------------
    # Public search methods
    # ------------------------------------------------------------------

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
            
            # Base query (without site restriction – phased search adds it)
            base_query = f'"{company_name}" investor relations official website'

            organic_results = self._phased_search(base_query, num=10)
            logger.info(f"[WEB_SEARCH] Filtered to {len(organic_results)} result(s)")

            if not organic_results:
                logger.warning(f"[WEB_SEARCH] No results after phased search for {company_name}")
                return {
                    "official_domain": None,
                    "investor_url": None,
                    "search_results": [],
                    "success": False,
                    "error": "No results from priority or fallback domains"
                }
            
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
        
        Uses the two‑phase search (priority → fallback domains) and applies
        domain filtering from the centralised config.
        
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
            query_parts.append("filetype:pdf")

            base_query = " ".join(query_parts)

            # Phased search already restricts to priority / fallback domains
            organic_results = self._phased_search(base_query, num=10)

            # Additional PDF‑specific filtering
            filtered_results: List[Dict] = []
            for result in organic_results:
                link = result.get("link", "")
                domain = _extract_domain(link)

                # Blocked‑domain check (belt‑and‑suspenders – _phased_search
                # already filters, but guard against future refactors)
                if _is_blocked(domain):
                    logger.debug(
                        "[WEB_SEARCH][docs] Rejected blocked domain: %s (%s)",
                        domain, link,
                    )
                    continue

                # Only accept PDFs from priority or fallback domains
                if link.lower().endswith(".pdf") and (
                    _is_priority_domain(domain) or _is_fallback_domain(domain)
                ):
                    filtered_results.append({
                        "title": result.get("title", ""),
                        "link": link,
                        "snippet": result.get("snippet", ""),
                        "type": document_type,
                    })
                else:
                    logger.debug(
                        "[WEB_SEARCH][docs] Skipped non‑PDF or non‑allowed domain: %s (%s)",
                        domain, link,
                    )
            
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
