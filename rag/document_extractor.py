"""
Document Extractor for Web-Retrieved Documents

Extracts text and structured data from PDFs downloaded from web.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

logger = logging.getLogger(__name__)


class DocumentExtractor:
    """Extracts text from downloaded documents."""
    
    @staticmethod
    def extract_text(file_path: Path, metadata: Dict) -> Dict:
        """
        Extract text from document.
        
        Args:
            file_path: Path to document file
            metadata: Document metadata
            
        Returns:
            Dict with:
            - text: str (extracted text)
            - pages: int (number of pages)
            - success: bool
        """
        if not file_path.exists():
            return {
                "text": "",
                "pages": 0,
                "success": False,
                "error": "File not found"
            }
        
        file_ext = file_path.suffix.lower()
        
        if file_ext == ".pdf":
            return DocumentExtractor._extract_from_pdf(file_path, metadata)
        else:
            logger.warning(f"[EXTRACTOR] Unsupported file type: {file_ext}")
            return {
                "text": "",
                "pages": 0,
                "success": False,
                "error": f"Unsupported file type: {file_ext}"
            }
    
    @staticmethod
    def _extract_from_pdf(file_path: Path, metadata: Dict) -> Dict:
        """Extract text from PDF file."""
        if PdfReader is None:
            return {
                "text": "",
                "pages": 0,
                "success": False,
                "error": "pypdf not installed"
            }
        
        try:
            reader = PdfReader(str(file_path))
            pages = []
            
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text.strip():
                    pages.append(f"[Page {page_num}]\n{text}")
            
            full_text = "\n\n".join(pages)
            
            logger.info(f"[EXTRACTOR] Extracted {len(reader.pages)} pages from PDF")
            
            return {
                "text": full_text,
                "pages": len(reader.pages),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"[EXTRACTOR] Failed to extract PDF: {e}", exc_info=True)
            return {
                "text": "",
                "pages": 0,
                "success": False,
                "error": str(e)
            }
