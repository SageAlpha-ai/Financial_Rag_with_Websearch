"""
Query Router

Determines whether to use RAG (documents) or LLM fallback.
Computes confidence scores based on retrieved content.
"""

import re
from typing import List, Dict, Any, Optional


def extract_year_from_query(query: str) -> Optional[str]:
    """Extracts fiscal year from query."""
    patterns = [
        r'FY\s*(\d{4})',
        r'fiscal\s+year\s+(\d{4})',
        r'\b(20\d{2})\b',
        r'\b(19\d{2})\b',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            year = match.group(1)
            return f"FY{year}"
    
    return None


def extract_entity_from_query(query: str) -> Optional[str]:
    """Extracts company/entity from query."""
    query_lower = query.lower()
    
    entity_mappings = {
        "oracle financial services": "Oracle Financial Services Software Ltd",
        "oracle financial": "Oracle Financial Services Software Ltd",
        "ofss": "Oracle Financial Services Software Ltd",
        "microsoft": "Microsoft",
        "apple": "Apple",
        "google": "Google",
        "amazon": "Amazon",
        "meta": "Meta",
        "facebook": "Meta",
        "tesla": "Tesla",
        "nvidia": "NVIDIA",
    }
    
    for key, value in entity_mappings.items():
        if key in query_lower:
            return value
    
    return None


def extract_metrics_from_query(query: str) -> List[str]:
    """Extracts requested financial metrics."""
    metrics = []
    query_lower = query.lower()
    
    metric_mapping = {
        'revenue': ['revenue', 'sales', 'turnover'],
        'net_income': ['net income', 'net profit', 'profit', 'earnings', 'pat'],
        'ebitda': ['ebitda'],
        'gross_profit': ['gross profit'],
        'operating_income': ['operating income', 'operating profit', 'ebit'],
        'assets': ['assets', 'total assets'],
        'equity': ['equity'],
    }
    
    for metric_key, keywords in metric_mapping.items():
        if any(kw in query_lower for kw in keywords):
            metrics.append(metric_key)
    
    return metrics


def compute_rag_confidence(
    query: str,
    documents: List[str],
    metadatas: List[Dict]
) -> Dict[str, Any]:
    """
    Computes confidence score for RAG retrieval.
    
    Returns dict with:
    - use_rag: bool
    - reason: str
    - scores: various match counts
    """
    requested_year = extract_year_from_query(query)
    requested_entity = extract_entity_from_query(query)
    requested_metrics = extract_metrics_from_query(query)
    
    entity_matches = 0
    year_matches = 0
    metric_matches = 0
    strong_matches = 0
    
    for meta, doc in zip(metadatas, documents):
        doc_lower = doc.lower()
        
        # Entity match
        company = meta.get("company", "")
        entity_match = False
        
        if requested_entity and company:
            if requested_entity.lower() in company.lower() or company.lower() in requested_entity.lower():
                entity_matches += 1
                entity_match = True
        elif company:
            entity_matches += 1
            entity_match = True
        
        # Year match
        doc_year = meta.get("fiscal_year", "")
        year_match = False
        
        if requested_year and doc_year:
            if requested_year.lower() == doc_year.lower():
                year_matches += 1
                year_match = True
        
        # Metric match
        for metric in requested_metrics:
            if metric == "revenue" and "revenue" in doc_lower:
                metric_matches += 1
            elif metric == "net_income" and ("net income" in doc_lower or "net profit" in doc_lower):
                metric_matches += 1
            elif metric in doc_lower:
                metric_matches += 1
        
        # Strong match = entity + year
        if entity_match and year_match:
            strong_matches += 1
    
    # Compute score
    total_score = strong_matches * 3 + min(entity_matches, 3) + year_matches * 2 + metric_matches
    
    # Determine routing
    use_rag = False
    reason = ""
    
    if requested_entity and requested_year:
        if strong_matches >= 1:
            use_rag = True
            reason = f"Found {strong_matches} chunks with {requested_entity} + {requested_year}"
        else:
            reason = f"No documents found for {requested_entity} in {requested_year}"
    elif requested_year:
        if year_matches >= 1 and entity_matches >= 1:
            use_rag = True
            reason = f"Found {year_matches} chunks with {requested_year}"
        else:
            reason = f"No chunks found with {requested_year}"
    elif requested_entity:
        if entity_matches >= 2 and metric_matches >= 1:
            use_rag = True
            reason = f"Found {entity_matches} chunks for {requested_entity}"
        else:
            reason = f"Insufficient data for {requested_entity}"
    else:
        if entity_matches >= 2 and metric_matches >= 1:
            use_rag = True
            reason = f"Found {entity_matches} entity matches and {metric_matches} metric matches"
        else:
            reason = "Insufficient matches in retrieved documents"
    
    return {
        "use_rag": use_rag,
        "reason": reason,
        "requested_year": requested_year,
        "requested_entity": requested_entity,
        "requested_metrics": requested_metrics,
        "entity_matches": entity_matches,
        "year_matches": year_matches,
        "metric_matches": metric_matches,
        "strong_matches": strong_matches,
        "total_score": total_score
    }


def route_query(
    query: str,
    documents: List[str],
    metadatas: List[Dict]
) -> tuple[bool, Dict[str, Any]]:
    """
    Routes query to RAG or LLM fallback.
    
    Returns:
        Tuple of (use_rag: bool, confidence: dict)
    """
    confidence = compute_rag_confidence(query, documents, metadatas)
    
    if confidence["use_rag"]:
        print(f"[Routing: RAG (confidence={confidence['total_score']}, {confidence['reason']})]")
    else:
        print(f"[Routing: LLM FALLBACK (confidence={confidence['total_score']}, {confidence['reason']})]")
        if confidence["requested_year"]:
            print(f"[Note: Requested year {confidence['requested_year']} not found in documents]")
    
    return confidence["use_rag"], confidence
