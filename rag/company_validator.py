"""
Company Name Validator

Validates that retrieved documents match the company mentioned in the query.
Prevents cross-company contamination.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CompanyValidator:
    """Validates company name matching to prevent cross-company contamination."""
    
    @staticmethod
    def validate_company_match(
        query: str,
        documents: List[str],
        metadatas: List[Dict]
    ) -> Tuple[List[str], List[Dict], List[str]]:
        """
        Filter documents to only include those matching the company in query.
        
        Args:
            query: User query
            documents: Retrieved documents
            metadatas: Document metadata
            
        Returns:
            (filtered_documents, filtered_metadatas, rejected_reasons)
        """
        query_company = CompanyValidator._extract_company_from_query(query)
        
        if not query_company:
            # No company specified, return all documents
            return documents, metadatas, []
        
        filtered_docs = []
        filtered_metas = []
        rejected = []
        
        for doc, meta in zip(documents, metadatas):
            doc_company = CompanyValidator._extract_company_from_metadata(meta, doc)
            
            if CompanyValidator._companies_match(query_company, doc_company):
                filtered_docs.append(doc)
                filtered_metas.append(meta)
            else:
                rejected.append(f"Document company '{doc_company}' doesn't match query company '{query_company}'")
                logger.warning(f"[VALIDATOR] Rejected document: {doc_company} != {query_company}")
        
        if rejected:
            logger.info(f"[VALIDATOR] Filtered {len(rejected)} documents due to company mismatch")
        
        return filtered_docs, filtered_metas, rejected
    
    @staticmethod
    def _extract_company_from_query(query: str) -> Optional[str]:
        """Extract company name from query."""
        query_lower = query.lower()
        
        # Check entity mappings FIRST (most reliable)
        entity_mappings = {
            "oracle financial services software": "Oracle Financial Services Software Ltd",
            "oracle financial services": "Oracle Financial Services Software Ltd",
            "oracle financial": "Oracle Financial Services Software Ltd",
            "ofss": "Oracle Financial Services Software Ltd",
            "infosys limited": "Infosys Limited",
            "infosys": "Infosys Limited",
            "tcs": "Tata Consultancy Services Limited",
            "tata consultancy services": "Tata Consultancy Services Limited",
            "tata consultancy": "Tata Consultancy Services Limited",
            "wipro limited": "Wipro Limited",
            "wipro": "Wipro Limited",
            "hcl technologies": "HCL Technologies Limited",
            "hcl": "HCL Technologies Limited",
            "icici bank": "ICICI Bank Limited",
            "icici": "ICICI Bank Limited",
            "icici bank limited": "ICICI Bank Limited",
            "hdfc bank": "HDFC Bank Limited",
            "hdfc": "HDFC Bank Limited",
            "axis bank": "Axis Bank Limited",
            "axis": "Axis Bank Limited",
            "sbi": "State Bank of India",
            "state bank of india": "State Bank of India",
        }
        
        # Check for full company names first (longest match wins)
        for key, value in sorted(entity_mappings.items(), key=len, reverse=True):
            if key in query_lower:
                logger.info(f"[VALIDATOR] Extracted company from query: '{value}' (matched '{key}')")
                return value
        
        # Common company name patterns (fallback)
        company_patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:Limited|Ltd|Inc|Corporation|Corp|LLC)\b',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Financial|Services|Technologies|Tech|Bank)\s+(?:Software|Limited|Ltd)?)\b',
            # Bank-specific patterns (handle "ICICI Bank", "HDFC Bank", etc.)
            r'\b([A-Z]+(?:\s+[A-Z]+)*)\s+Bank\b',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+Bank\b',
        ]
        
        for pattern in company_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                # Check if it's a known bank and normalize
                extracted_lower = extracted.lower()
                if "icici" in extracted_lower:
                    logger.info(f"[VALIDATOR] Extracted company from pattern: 'ICICI Bank Limited' (matched '{extracted}')")
                    return "ICICI Bank Limited"
                elif "hdfc" in extracted_lower:
                    logger.info(f"[VALIDATOR] Extracted company from pattern: 'HDFC Bank Limited' (matched '{extracted}')")
                    return "HDFC Bank Limited"
                elif "axis" in extracted_lower:
                    logger.info(f"[VALIDATOR] Extracted company from pattern: 'Axis Bank Limited' (matched '{extracted}')")
                    return "Axis Bank Limited"
                elif "sbi" in extracted_lower or "state bank" in extracted_lower:
                    logger.info(f"[VALIDATOR] Extracted company from pattern: 'State Bank of India' (matched '{extracted}')")
                    return "State Bank of India"
                else:
                    logger.info(f"[VALIDATOR] Extracted company from pattern: '{extracted}'")
                    return extracted
        
        logger.warning(f"[VALIDATOR] Could not extract company name from query: {query}")
        return None
    
    @staticmethod
    def _extract_company_from_metadata(meta: Dict, doc_text: str = "") -> Optional[str]:
        """Extract company name from document metadata or text."""
        # Try metadata first
        company = meta.get("company") or meta.get("entity") or meta.get("organization")
        if company:
            return str(company).strip()
        
        # Try to extract from filename
        filename = meta.get("filename", meta.get("source", ""))
        if filename:
            # Remove path and extension
            filename = filename.split("/")[-1].split("\\")[-1]
            filename = re.sub(r'\.[^.]+$', '', filename)
            
            # Check if it contains company name patterns
            company_match = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', filename)
            if company_match:
                return company_match.group(1).strip()
        
        # Try to extract from document text (first few lines)
        if doc_text:
            first_lines = "\n".join(doc_text.split("\n")[:10])
            company_match = re.search(
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:Limited|Ltd|Inc)',
                first_lines
            )
            if company_match:
                return company_match.group(1).strip()
        
        return None
    
    @staticmethod
    def _companies_match(company1: str, company2: Optional[str]) -> bool:
        """Check if two company names match (fuzzy matching)."""
        if not company1 or not company2:
            # If one is missing, don't filter (be permissive)
            logger.debug(f"[VALIDATOR] One company missing: '{company1}' vs '{company2}' - allowing match")
            return True
        
        # Normalize company names
        def normalize(name: str) -> str:
            name = name.lower().strip()
            # Remove common suffixes
            name = re.sub(r'\s+(limited|ltd|inc|incorporated|corp|corporation|llc|software)\s*$', '', name)
            # Remove common prefixes
            name = re.sub(r'^(the\s+)', '', name)
            # Remove extra spaces
            name = re.sub(r'\s+', ' ', name)
            return name.strip()
        
        norm1 = normalize(company1)
        norm2 = normalize(company2)
        
        logger.debug(f"[VALIDATOR] Comparing: '{norm1}' vs '{norm2}'")
        
        # Exact match
        if norm1 == norm2:
            logger.debug(f"[VALIDATOR] Exact match: '{norm1}' == '{norm2}'")
            return True
        
        # Check if one contains the other (for partial matches)
        # This handles "Oracle Financial Services" matching "Oracle Financial Services Software Ltd"
        if norm1 in norm2 or norm2 in norm1:
            logger.debug(f"[VALIDATOR] Partial match: '{norm1}' in '{norm2}' or vice versa")
            return True
        
        # Check for common abbreviations and aliases
        abbreviations = {
            "oracle financial services software": ["oracle financial services", "oracle financial", "ofss"],
            "oracle financial services": ["oracle financial", "ofss"],
            "tata consultancy services": ["tcs", "tata consultancy"],
            "infosys limited": ["infosys"],
            "wipro limited": ["wipro"],
            "hcl technologies": ["hcl"],
        }
        
        # Check if both companies are in the same alias group
        for full_name, abbrevs in abbreviations.items():
            all_variants = [full_name] + abbrevs
            if norm1 in all_variants and norm2 in all_variants:
                logger.debug(f"[VALIDATOR] Alias match: '{norm1}' and '{norm2}' are variants of '{full_name}'")
                return True
        
        # Token-based matching (check if key words match)
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        
        # If at least 2 key words match, consider it a match
        common_words = words1.intersection(words2)
        if len(common_words) >= 2:
            logger.debug(f"[VALIDATOR] Token match: {len(common_words)} common words: {common_words}")
            return True
        
        logger.debug(f"[VALIDATOR] No match: '{norm1}' != '{norm2}'")
        return False
