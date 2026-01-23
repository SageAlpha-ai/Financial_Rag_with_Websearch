"""
Query Engine

Wraps all RAG logic into a single answer_query function.
Used by both CLI (main.py) and API (api.py).
"""

import re
from typing import Dict, Any, List, Optional

from rag.retriever import retrieve_with_year_filter
from rag.router import route_query
from rag.answer_formatter import (
    format_rag_response,
    format_llm_fallback_response,
    format_sources,
    check_entity_confidence
)


def validate_rag_answer(answer: str, query: str) -> tuple[bool, str]:
    """
    Validates if a RAG answer actually contains useful information.
    
    Returns:
        Tuple of (is_valid: bool, reason: str)
    """
    answer_lower = answer.lower()
    
    # Failure phrases that indicate RAG didn't find the answer
    # NEVER return "Not available" to the user
    failure_phrases = [
        "not available",
        "not found",
        "not explicitly stated",
        "not provided in the document",
        "missing from the document",
        "missing in the retrieved documents",
        "the document does not provide",
        "not found in the context",
        "not mentioned in the document",
        "no information available",
        "cannot be determined from",
        "not specified in the",
        "does not contain",
        "not present in",
        "information is not",
        "data is not",
    ]
    
    for phrase in failure_phrases:
        if phrase in answer_lower:
            return (False, f"Answer contains '{phrase}'")
    
    # Check if answer is too short (likely unhelpful)
    content_lines = [
        line.strip() for line in answer.split('\n')
        if line.strip() and not line.startswith('-') and not line.startswith('=')
    ]
    content_text = ' '.join(content_lines)
    
    if len(content_text) < 50:
        return (False, "Answer too short to be useful")
    
    # For financial queries, check if key data is present
    financial_keywords = ["revenue", "income", "profit", "employee", "asset", "earning"]
    is_financial_query = any(kw in query.lower() for kw in financial_keywords)
    
    if is_financial_query:
        has_numbers = bool(re.search(r'\d+[,.]?\d*', answer))
        if not has_numbers:
            return (False, "Financial query but no numbers in answer")
    
    return (True, "Answer validated successfully")


def answer_query(question: str) -> Dict[str, Any]:
    """
    Main query answering function.
    
    Retrieves documents, routes to RAG or LLM,
    and returns structured response.
    
    Args:
        question: User question string
    
    Returns:
        Dict with keys:
        - answer: str
        - answer_type: "RAG" | "LLM"
        - sources: List[str] | None
        - metadata: dict (optional debug info)
    """
    # Step 1: Retrieve documents
    documents, metadatas, requested_year = retrieve_with_year_filter(question)
    
    # Step 2: Check entity confidence
    entity, entity_count = check_entity_confidence(metadatas)
    
    # Step 3: Route query
    use_rag, confidence = route_query(question, documents, metadatas)
    
    # Step 4: Generate response
    sources: Optional[List[str]] = None
    answer_type = "LLM"
    
    if use_rag:
        # === RAG PATH ===
        answer = format_rag_response(question, documents[:5], metadatas[:5])
        
        # === POST-RAG VALIDATION ===
        is_valid, validation_reason = validate_rag_answer(answer, question)
        
        if is_valid:
            # RAG answer is good
            answer_type = "RAG"
            
            if entity_count < 2 and entity_count > 0:
                answer += "\n\n---\nENTITY CONFIDENCE: Low (based on limited document confirmation)"
            
            # Extract sources
            sources = []
            for meta in metadatas[:5]:
                source = meta.get('source', meta.get('filename', 'unknown'))
                if 'fiscal_year' in meta and meta['fiscal_year'] != 'Unknown':
                    source += f" (FY: {meta['fiscal_year']})"
                sources.append(source)
        else:
            # RAG failed - fallback to LLM
            answer = format_llm_fallback_response(question)
            answer_type = "LLM"
            sources = None
    else:
        # === LLM FALLBACK PATH ===
        answer = format_llm_fallback_response(question)
        answer_type = "LLM"
        sources = None
    
    return {
        "answer": answer,
        "answer_type": answer_type,
        "sources": sources,
        "metadata": {
            "entity_detected": entity,
            "entity_confidence": entity_count,
            "requested_year": requested_year,
            "rag_confidence_score": confidence.get("total_score", 0)
        }
    }


def answer_query_simple(question: str) -> Dict[str, Any]:
    """
    Simplified version for API responses.
    Returns only answer, answer_type, and sources.
    """
    result = answer_query(question)
    return {
        "answer": result["answer"],
        "answer_type": result["answer_type"],
        "sources": result["sources"]
    }
