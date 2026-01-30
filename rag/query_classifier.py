"""
Query Classifier for SageAlpha AI

Classifies queries to determine if they require verified sources.
"""

import logging
import re
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class QueryClassifier:
    """Classifies queries to determine evidence requirements and intent."""
    
    # Greeting patterns
    GREETING_PATTERNS = [
        r'^(hi|hello|hey|greetings|good morning|good afternoon|good evening|howdy)[\s!.,]*$',
        r'^(hi|hello|hey)[\s,]+(there|everyone|all|guys)[\s!.,]*$',
        r'^how\s+(are\s+you|do\s+you\s+do|is\s+it\s+going)[\s?.,]*$',
        r'^what\'?s\s+up[\s?.,]*$',
        r'^thanks?[\s!.,]*$',
        r'^thank\s+you[\s!.,]*$',
    ]
    
    # General knowledge indicators (not document-specific)
    GENERAL_KNOWLEDGE_INDICATORS = [
        "what is", "who is", "explain", "tell me about", "describe",
        "how does", "how do", "what are", "define", "meaning of",
        "difference between", "compare", "why is", "why are"
    ]
    
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
        Classify query to determine intent and evidence requirements.
        
        Returns:
            {
                "intent": "greeting" | "general_knowledge" | "financial_document_query",
                "is_factual_financial": bool,
                "requires_verified_source": bool,
                "confidence": float,
                "keywords_found": List[str]
            }
        """
        query_lower = query.lower().strip()
        query_clean = re.sub(r'[^\w\s]', '', query_lower)
        
        # STEP 1: Check for greetings (highest priority)
        for pattern in QueryClassifier.GREETING_PATTERNS:
            if re.match(pattern, query_lower, re.IGNORECASE):
                logger.info(f"[CLASSIFIER] Intent: greeting")
                return {
                    "intent": "greeting",
                    "is_factual_financial": False,
                    "requires_verified_source": False,
                    "confidence": 1.0,
                    "keywords_found": []
                }
        
        # STEP 2: Check for financial keywords
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
        
        # STEP 3: Determine intent
        is_factual_financial = len(keywords_found) > 0 or has_numeric
        
        # Check if it's general knowledge (not document-specific)
        is_general_knowledge = False
        if not is_factual_financial:
            for indicator in QueryClassifier.GENERAL_KNOWLEDGE_INDICATORS:
                if indicator in query_lower:
                    is_general_knowledge = True
                    break
        
        # Determine intent
        if is_factual_financial:
            intent = "financial_document_query"
        elif is_general_knowledge:
            intent = "general_knowledge"
        else:
            # Default to financial_document_query if unclear (safer to use RAG)
            intent = "financial_document_query"
        
        # Calculate confidence
        confidence = 0.0
        if len(keywords_found) >= 2:
            confidence = 1.0
        elif len(keywords_found) == 1:
            confidence = 0.7
        elif has_numeric:
            confidence = 0.5
        elif is_general_knowledge:
            confidence = 0.8
        
        requires_verified_source = is_factual_financial
        
        logger.info(f"[CLASSIFIER] Intent: {intent}, factual_financial={is_factual_financial}, "
                   f"requires_source={requires_verified_source}, confidence={confidence:.2f}")
        if keywords_found:
            logger.info(f"[CLASSIFIER] Keywords found: {keywords_found}")
        
        return {
            "intent": intent,
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
