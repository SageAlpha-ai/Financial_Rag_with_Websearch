"""
Source Formatter for SageAlpha AI

Formats sources to hide internal paths and expose only official, user-facing URLs.
"""

import logging
import re
from typing import Dict, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class SourceFormatter:
    """
    Formats sources for user-facing responses.
    
    Rules:
    - Hide Azure Blob paths
    - Hide temp storage paths
    - Show official investor relations URLs
    - Show official regulator URLs
    - Mask internal filenames
    """
    
    @staticmethod
    def format_sources(
        metadatas: List[Dict],
        web_documents: List[Dict] = None
    ) -> List[Dict]:
        """
        Format sources for user-facing response.
        
        CRITICAL: Always returns at least one source if metadatas are provided.
        Never returns empty list for RAG answers.
        
        Returns:
            List of source dicts with:
            - title: str
            - url: str (official URL only, optional)
            - publisher: str
            - year: str (optional)
            - page: int (optional)
            - note: str (optional)
        """
        sources = []
        
        # Format RAG sources (internal documents)
        # CRITICAL: Always process all metadatas, never skip
        for meta in metadatas:
            source = SourceFormatter._format_rag_source(meta)
            # _format_rag_source now always returns a dict (never None)
            if source:
                sources.append(source)
        
        # Format web sources (official documents)
        if web_documents:
            for web_doc in web_documents:
                source = SourceFormatter._format_web_source(web_doc.get("metadata", {}))
                if source:
                    sources.append(source)
        
        # Deduplicate sources
        deduplicated = SourceFormatter._deduplicate_sources(sources)
        
        # CRITICAL: If we had metadatas but got empty sources, create fallback
        if not deduplicated and metadatas:
            logger.warning("[SOURCE] Formatter returned empty after deduplication, creating fallback")
            deduplicated = SourceFormatter._create_fallback_sources(metadatas)
        
        return deduplicated
    
    @staticmethod
    def _format_rag_source(meta: Dict) -> Optional[Dict]:
        """
        Format internal RAG source to user-facing format.
        
        CRITICAL: Always returns a source dict, never None.
        Uses fallback logic to ensure sources are always populated.
        """
        source_path = meta.get("source", meta.get("filename", ""))
        
        # Extract document title (with comprehensive fallback)
        company = meta.get("company", "")
        fiscal_year = meta.get("fiscal_year", "")
        doc_type = meta.get("document_type", "")
        
        # Strategy 1: Try to extract from existing title fields
        title = SourceFormatter._extract_document_title(meta)
        
        # Strategy 2: If title is empty or just contains year, build from metadata
        if not title or title == "Document" or title.strip() == f"({fiscal_year})" or (fiscal_year and title.strip() == fiscal_year):
            # Build title from company + year + type
            parts = []
            if company:
                parts.append(company)
            
            # Add document type
            if doc_type and doc_type != "other":
                type_name = doc_type.replace("_", " ").title()
                if "annual" in type_name.lower():
                    parts.append("Annual Report")
                elif "financial" in type_name.lower():
                    parts.append("Financial Statements")
                else:
                    parts.append(type_name)
            else:
                parts.append("Financial Statements")
            
            if fiscal_year:
                parts.append(f"({fiscal_year})")
            
            if parts:
                # Join with proper separator, avoiding double dashes
                title = " – ".join(parts)
                # Clean up any double dashes
                title = re.sub(r'\s*–\s*–\s*', ' – ', title)
            else:
                title = "Financial Document from SageAlpha Knowledge Base"
        
        # Strategy 3: Ensure title is meaningful (final fallback)
        if not title or len(title.strip()) < 10:
            if company and fiscal_year:
                title = f"{company} – {fiscal_year} Financial Statements"
            elif company:
                title = f"{company} – Financial Documents"
            elif fiscal_year:
                title = f"Financial Documents ({fiscal_year})"
            else:
                title = "Financial Document from SageAlpha Knowledge Base"
        
        # Determine publisher
        company = meta.get("company", "")
        if company:
            publisher = company
        elif SourceFormatter._is_regulatory_document(meta):
            publisher = "SEBI" if "sebi" in str(meta).lower() else "Regulator"
        else:
            publisher = "SageAlpha Knowledge Base"
        
        # Determine URL
        url = None
        
        # If it's already a valid URL, use it
        if source_path.startswith("http"):
            parsed = urlparse(source_path)
            if SourceFormatter._is_official_domain(parsed.netloc):
                url = source_path
        
        # For regulatory documents, try to construct official URL
        if not url and SourceFormatter._is_regulatory_document(meta):
            url = SourceFormatter._get_regulatory_url(meta)
        
        # For company documents, try to extract or construct investor relations URL
        if not url and company:
            # Check if metadata has official URL
            official_url = meta.get("official_url") or meta.get("investor_url")
            if official_url:
                url = official_url
        
        # Build source dict (always return something)
        source = {
            "title": title,
            "publisher": publisher,
            "url": url
        }
        
        # Add year if available
        if meta.get("fiscal_year"):
            source["year"] = meta.get("fiscal_year")
        
        # Add page if available
        if meta.get("page"):
            source["page"] = meta.get("page")
        
        # Add note for internal documents
        if SourceFormatter._is_internal_path(source_path) or not url:
            source["note"] = "Retrieved from SageAlpha Knowledge Base (derived from official company filings)"
        
        return source
    
    @staticmethod
    def _format_web_source(meta: Dict) -> Optional[Dict]:
        """Format web-retrieved source to user-facing format."""
        url = meta.get("url", "")
        title = meta.get("title", "Official Document")
        
        if not url:
            return None
        
        # Parse URL to get official domain
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        # If it's a CDN or third-party hosting, try to find investor relations root
        if SourceFormatter._is_cdn_url(domain):
            # Try to extract company domain and construct investor relations URL
            investor_url = SourceFormatter._get_investor_relations_url(meta)
            if investor_url:
                url = investor_url
                domain = urlparse(investor_url).netloc.lower()
        
        # Extract company name from metadata or URL
        company_name = meta.get("company") or SourceFormatter._extract_company_from_url(url)
        
        return {
            "title": f"{company_name or 'Company'} – Investor Relations" if "investor" in url.lower() or "ir" in url.lower() else title,
            "url": url,
            "publisher": company_name or "Company Official Website"
        }
    
    @staticmethod
    def _is_internal_path(path: str) -> bool:
        """Check if path is an internal path that should be hidden."""
        if not path:
            return True
        
        internal_indicators = [
            "azure_blob",
            "blob.core.windows.net",
            "temp",
            "tmp",
            "rag_web_docs",
            "documents/",
            ".pdf",
            "\\",  # Windows path
            "/",   # Unix path (if not URL)
        ]
        
        # If it's a URL, it's not internal
        if path.startswith("http"):
            return False
        
        # Check for internal indicators
        path_lower = path.lower()
        return any(indicator in path_lower for indicator in internal_indicators)
    
    @staticmethod
    def _is_cdn_url(domain: str) -> bool:
        """Check if domain is a CDN or third-party hosting."""
        cdn_domains = [
            "cloudfront",
            "amazonaws",
            "q4cdn",
            "nasdaq",
            "sec.gov",  # SEC is official but we want company site
            "edgar",
        ]
        return any(cdn in domain for cdn in cdn_domains)
    
    @staticmethod
    def _is_official_domain(domain: str) -> bool:
        """Check if domain is an official company or regulator domain."""
        # Reject CDNs and third-party hosts
        if SourceFormatter._is_cdn_url(domain):
            return False
        
        # Accept company domains and official regulators
        official_indicators = [
            ".com",
            ".in",
            ".org",
            "sebi.gov.in",
            "mca.gov.in",
        ]
        
        return any(indicator in domain for indicator in official_indicators)
    
    @staticmethod
    def _extract_document_title(meta: Dict) -> str:
        """Extract user-friendly document title from metadata."""
        # Try various fields
        title = meta.get("title") or meta.get("document_title") or meta.get("filename", "")
        
        # Clean up filename
        if title and "/" in title:
            title = title.split("/")[-1]
        if title and "\\" in title:
            title = title.split("\\")[-1]
        
        # Remove extension
        title = re.sub(r'\.[^.]+$', '', title)
        
        # Add fiscal year if available
        if meta.get("fiscal_year"):
            title += f" ({meta.get('fiscal_year')})"
        
        return title or "Document"
    
    @staticmethod
    def _is_regulatory_document(meta: Dict) -> bool:
        """Check if document is a regulatory document (SEBI, etc.)."""
        source = str(meta.get("source", "")).lower()
        filename = str(meta.get("filename", "")).lower()
        
        regulatory_keywords = ["sebi", "regulation", "circular", "guideline", "act"]
        return any(kw in source or kw in filename for kw in regulatory_keywords)
    
    @staticmethod
    def _get_regulatory_url(meta: Dict) -> Optional[str]:
        """Construct official regulatory URL if possible."""
        # For SEBI documents, try to construct official URL
        if "sebi" in str(meta.get("source", "")).lower():
            # SEBI official website
            return "https://www.sebi.gov.in"
        
        return None
    
    @staticmethod
    def _extract_publisher(meta: Dict) -> str:
        """Extract publisher name from metadata."""
        if meta.get("company"):
            return meta.get("company")
        if "sebi" in str(meta.get("source", "")).lower():
            return "SEBI"
        return "Regulator"
    
    @staticmethod
    def _get_investor_relations_url(meta: Dict) -> Optional[str]:
        """Get investor relations root URL from metadata."""
        investor_url = meta.get("investor_url") or meta.get("investor_root")
        if investor_url:
            return investor_url
        
        # Try to construct from company domain
        company_domain = meta.get("official_domain")
        if company_domain:
            # Try common investor relations paths
            for path in ["/investor-relations", "/investors", "/investor", "/ir"]:
                test_url = f"https://{company_domain}{path}"
                # In production, you might want to verify this URL exists
                return test_url
        
        return None
    
    @staticmethod
    def _extract_company_from_url(url: str) -> Optional[str]:
        """Extract company name from URL."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Remove www. and common TLDs
            domain = re.sub(r'^www\.', '', domain)
            domain = re.sub(r'\.(com|in|org|net)$', '', domain)
            
            # Extract company name (first part before any subdomain)
            parts = domain.split('.')
            if parts:
                company = parts[0].replace('-', ' ').title()
                return company
            
        except:
            pass
        
        return None
    
    @staticmethod
    def _deduplicate_sources(sources: List[Dict]) -> List[Dict]:
        """Remove duplicate sources based on URL."""
        seen_urls = set()
        unique_sources = []
        
        for source in sources:
            url = source.get("url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_sources.append(source)
            elif not url:
                # Sources without URLs are unique by title
                title = source.get("title", "")
                if title not in [s.get("title", "") for s in unique_sources]:
                    unique_sources.append(source)
        
        return unique_sources
