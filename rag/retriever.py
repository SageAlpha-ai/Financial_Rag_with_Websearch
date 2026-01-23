"""
Document Retriever

Queries Chroma Cloud for relevant documents.
Supports year-aware filtering for financial queries.
"""

import re
from typing import List, Dict, Optional, Tuple

from vectorstore.chroma_client import get_collection


def extract_year_from_query(query: str) -> Optional[str]:
    """
    Extracts fiscal year from query.
    Returns normalized FYxxxx format.
    """
    patterns = [
        r'FY\s*(\d{4})',           # FY2023
        r'fiscal\s+year\s+(\d{4})', # fiscal year 2023
        r'\b(20\d{2})\b',          # 2023
        r'\b(19\d{2})\b',          # 1999
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            year = match.group(1)
            return f"FY{year}"
    
    return None


def retrieve_documents(
    query: str,
    n_results: int = 10,
    collection_name: str = None
) -> Tuple[List[str], List[Dict]]:
    """
    Retrieves documents from Chroma Cloud.
    
    Args:
        query: User query text
        n_results: Number of results to return
        collection_name: Target collection
    
    Returns:
        Tuple of (documents, metadatas)
    """
    collection = get_collection(collection_name, create_if_missing=False)
    
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas"]
    )
    
    documents = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    
    return documents, metadatas


def retrieve_with_year_filter(
    query: str,
    n_results: int = 10,
    collection_name: str = None
) -> Tuple[List[str], List[Dict], Optional[str]]:
    """
    Retrieves documents with year-aware filtering.
    
    If query contains a year, first tries to retrieve year-specific
    documents, then supplements with general results.
    
    Args:
        query: User query text
        n_results: Number of results to return
        collection_name: Target collection
    
    Returns:
        Tuple of (documents, metadatas, requested_year)
    """
    collection = get_collection(collection_name, create_if_missing=False)
    requested_year = extract_year_from_query(query)
    
    if requested_year:
        # Try year-filtered retrieval first
        try:
            year_results = collection.query(
                query_texts=[query],
                n_results=5,
                where={"fiscal_year": requested_year},
                include=["documents", "metadatas"]
            )
            year_documents = year_results["documents"][0] if year_results["documents"] else []
            year_metadatas = year_results["metadatas"][0] if year_results["metadatas"] else []
        except Exception:
            year_documents = []
            year_metadatas = []
        
        # Get general results too
        general_results = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas"]
        )
        general_documents = general_results["documents"][0] if general_results["documents"] else []
        general_metadatas = general_results["metadatas"][0] if general_results["metadatas"] else []
        
        # Combine: year-specific first
        if year_documents:
            # Deduplicate by avoiding docs already in year_documents
            additional_docs = []
            additional_metas = []
            for doc, meta in zip(general_documents, general_metadatas):
                if doc not in year_documents:
                    additional_docs.append(doc)
                    additional_metas.append(meta)
            
            documents = year_documents + additional_docs[:5]
            metadatas = year_metadatas + additional_metas[:5]
            
            print(f"[Year-aware retrieval: Found {len(year_documents)} chunks for {requested_year}]")
        else:
            documents = general_documents
            metadatas = general_metadatas
            print(f"[Year-aware retrieval: No chunks found for {requested_year}]")
    else:
        # No year specified, general retrieval
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas"]
        )
        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
    
    return documents, metadatas, requested_year
