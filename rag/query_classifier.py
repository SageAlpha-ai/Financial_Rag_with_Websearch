"""
Query Classifier for SageAlpha AI

Classifies queries to determine if they require verified sources.
"""

import logging
import re
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class QueryClassifier:
    """Classifies queries to determine evidence requirements."""
    
    # Financial keywords that require verified sources
    FINANCIAL_KEYWORDS = [
        "revenue", "profit", "turnover", "income", "earnings",
        "financial year", "fy", "fiscal year", "annual report",
        "balance sheet", "cash flow", "assets", "liabilities",
        "dividend", "eps", "pe ratio", "market cap", "valuation",
        "sales", "expenses", "cost", "margin", "ebitda", "ebit",
        "net income", "gross profit", "operating profit",
        "quarterly", "q1", "q2", "q3", "q4", "10-k", "10-q",
        "sec filing", "financial statement", "audit", "compliance"
    ]
    
    # Numeric patterns that indicate factual queries
    NUMERIC_PATTERNS = [
        r'\d+[,\d]*\s*(crore|million|billion|thousand|lakh)',
        r'₹\s*\d+',
        r'\$\s*\d+',
        r'\d+\s*%',
        r'fy\d{4}',
        r'\d{4}',
    ]
    
    @staticmethod
    def classify_query(query: str) -> Dict[str, any]:
        """
        Classify query to determine evidence requirements.
        
        Returns:
            {
                "is_factual_financial": bool,
                "requires_verified_source": bool,
                "confidence": float,
                "keywords_found": List[str]
            }
        """
        query_lower = query.lower()
        
        # Check for financial keywords
        keywords_found = []
        for keyword in QueryClassifier.FINANCIAL_KEYWORDS:
            if keyword in query_lower:
                keywords_found.append(keyword)
        
        # Check for numeric patterns
        has_numeric = False
        for pattern in QueryClassifier.NUMERIC_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                has_numeric = True
                break
        
        # Determine if factual financial query
        is_factual_financial = len(keywords_found) > 0 or has_numeric
        
        # Calculate confidence
        confidence = 0.0
        if len(keywords_found) >= 2:
            confidence = 1.0
        elif len(keywords_found) == 1:
            confidence = 0.7
        elif has_numeric:
            confidence = 0.5
        
        requires_verified_source = is_factual_financial
        
        logger.info(f"[CLASSIFIER] Query classified: factual_financial={is_factual_financial}, "
                   f"requires_source={requires_verified_source}, confidence={confidence:.2f}")
        if keywords_found:
            logger.info(f"[CLASSIFIER] Keywords found: {keywords_found}")
        
        return {
            "is_factual_financial": is_factual_financial,
            "requires_verified_source": requires_verified_source,
            "confidence": confidence,
            "keywords_found": keywords_found
        }
    
    @staticmethod
    def requires_verified_source(query: str) -> bool:
        """Quick check if query requires verified source."""
        classification = QueryClassifier.classify_query(query)
        return classification["requires_verified_source"]
