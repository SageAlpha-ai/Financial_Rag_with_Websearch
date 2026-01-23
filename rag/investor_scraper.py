"""
Investor Relations Scraper

Scrapes official investor relations pages to find and download financial documents.
Handles PDF downloads, document extraction, and metadata collection.
"""

import logging
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class InvestorRelationsScraper:
    """
    Scraper for company investor relations pages.
    
    Extracts:
    - Annual Reports
    - Financial Statements
    - SEC Filings (10-K, 10-Q)
    - Earnings Releases
    - Investor Presentations
    """
    
    def __init__(self, timeout: int = 10):
        """
        Initialize scraper.
        
        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
    
    def scrape_investor_page(
        self, 
        investor_url: str
    ) -> Dict[str, any]:
        """
        Scrape investor relations page for document links.
        
        Args:
            investor_url: URL of investor relations page
            
        Returns:
            Dict with:
            - documents: List of document metadata
            - sections: Dict of section names to document lists
            - success: bool
        """
        try:
            logger.info(f"[SCRAPER] Scraping investor page: {investor_url}")
            
            response = self.session.get(investor_url, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all document links
            documents = self._extract_document_links(soup, investor_url)
            
            # Organize by section
            sections = self._organize_by_section(soup, documents)
            
            logger.info(f"[SCRAPER] Found {len(documents)} documents across {len(sections)} sections")
            
            return {
                "documents": documents,
                "sections": sections,
                "success": True,
                "base_url": investor_url
            }
            
        except Exception as e:
            logger.error(f"[SCRAPER] Failed to scrape investor page: {e}", exc_info=True)
            return {
                "documents": [],
                "sections": {},
                "success": False,
                "error": str(e)
            }
    
    def _extract_document_links(
        self, 
        soup: BeautifulSoup, 
        base_url: str
    ) -> List[Dict]:
        """Extract all document links from the page."""
        documents = []
        
        # Find all links
        links = soup.find_all('a', href=True)
        
        for link in links:
            href = link.get('href', '')
            text = link.get_text(strip=True)
            
            # Make absolute URL
            absolute_url = urljoin(base_url, href)
            
            # Check if it's a document link
            if self._is_document_link(absolute_url, text):
                doc_type = self._classify_document(absolute_url, text)
                
                documents.append({
                    "title": text or self._extract_title_from_url(absolute_url),
                    "url": absolute_url,
                    "type": doc_type,
                    "extension": self._get_file_extension(absolute_url)
                })
        
        return documents
    
    def _is_document_link(self, url: str, text: str) -> bool:
        """Check if link points to a document."""
        url_lower = url.lower()
        text_lower = text.lower()
        
        # Check file extensions
        doc_extensions = ['.pdf', '.xlsx', '.xls', '.doc', '.docx']
        if any(url_lower.endswith(ext) for ext in doc_extensions):
            return True
        
        # Check for document-related keywords in URL or text
        doc_keywords = [
            'annual-report', 'annual report', '10-k', '10-q', '10k', '10q',
            'earnings', 'financial', 'filing', 'sec', 'quarterly', 'yearly',
            'presentation', 'investor', 'ir'
        ]
        
        if any(kw in url_lower or kw in text_lower for kw in doc_keywords):
            return True
        
        return False
    
    def _classify_document(self, url: str, text: str) -> str:
        """Classify document type."""
        url_lower = url.lower()
        text_lower = text.lower()
        
        if 'annual-report' in url_lower or 'annual report' in text_lower:
            return "annual_report"
        elif '10-k' in url_lower or '10k' in url_lower:
            return "10-k"
        elif '10-q' in url_lower or '10q' in url_lower:
            return "10-q"
        elif 'earnings' in url_lower or 'earnings' in text_lower:
            return "earnings_release"
        elif 'presentation' in url_lower or 'presentation' in text_lower:
            return "investor_presentation"
        elif 'financial' in url_lower or 'financial' in text_lower:
            return "financial_statement"
        else:
            return "other"
    
    def _extract_title_from_url(self, url: str) -> str:
        """Extract title from URL if text is missing."""
        parsed = urlparse(url)
        filename = Path(parsed.path).stem
        # Clean up filename
        filename = filename.replace('-', ' ').replace('_', ' ')
        return filename
    
    def _get_file_extension(self, url: str) -> str:
        """Get file extension from URL."""
        parsed = urlparse(url)
        path = Path(parsed.path)
        return path.suffix.lower()
    
    def _organize_by_section(
        self, 
        soup: BeautifulSoup, 
        documents: List[Dict]
    ) -> Dict[str, List[Dict]]:
        """Organize documents by section (Overview, Financials, SEC filings, etc.)."""
        sections = {
            "Overview": [],
            "Financials": [],
            "SEC filings": [],
            "Investor news": [],
            "Events and presentations": [],
            "FAQ": [],
            "Earnings Release": [],
            "Financial Tables": [],
            "Customer Wins": [],
            "Webcast": [],
            "10-Q/K": [],
            "Recent investor news": [],
            "Additional resources": []
        }
        
        # Try to find section headers
        section_headers = soup.find_all(['h1', 'h2', 'h3', 'h4'], string=re.compile(
            r'(Overview|Financials|SEC|Filing|News|Event|Presentation|FAQ|Earnings|Webcast|10-?[QK])',
            re.IGNORECASE
        ))
        
        # For now, categorize by document type
        for doc in documents:
            doc_type = doc.get("type", "other")
            
            if doc_type == "annual_report":
                sections["Financials"].append(doc)
            elif doc_type in ["10-k", "10-q"]:
                sections["SEC filings"].append(doc)
            elif doc_type == "earnings_release":
                sections["Earnings Release"].append(doc)
            elif doc_type == "investor_presentation":
                sections["Events and presentations"].append(doc)
            else:
                sections["Additional resources"].append(doc)
        
        # Remove empty sections
        return {k: v for k, v in sections.items() if v}
    
    def download_document(
        self, 
        url: str, 
        save_path: Optional[Path] = None
    ) -> Tuple[Optional[Path], Dict]:
        """
        Download a document from URL.
        
        Args:
            url: Document URL
            save_path: Optional path to save file (if None, uses temp directory)
            
        Returns:
            (file_path, metadata_dict)
        """
        try:
            logger.info(f"[SCRAPER] Downloading document: {url}")
            
            response = self.session.get(url, timeout=self.timeout, stream=True)
            response.raise_for_status()
            
            # Determine save path
            if save_path is None:
                # Use temp directory
                temp_dir = Path(tempfile.gettempdir()) / "rag_web_docs"
                temp_dir.mkdir(exist_ok=True)
                
                # Extract filename from URL
                parsed = urlparse(url)
                filename = Path(parsed.path).name or "document.pdf"
                save_path = temp_dir / filename
            
            # Ensure parent directory exists
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download file
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Get metadata
            metadata = {
                "url": url,
                "filename": save_path.name,
                "file_path": str(save_path),
                "size": save_path.stat().st_size,
                "content_type": response.headers.get("Content-Type", ""),
                "source_type": "live_web",
                "publisher": "Company Official Website",
                "ephemeral": True
            }
            
            logger.info(f"[SCRAPER] Document downloaded: {save_path} ({metadata['size']} bytes)")
            
            return save_path, metadata
            
        except Exception as e:
            logger.error(f"[SCRAPER] Failed to download document: {e}", exc_info=True)
            return None, {"error": str(e)}
