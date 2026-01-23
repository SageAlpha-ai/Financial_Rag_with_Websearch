"""RAG module - Retrieval, routing, and answer formatting."""
from .retriever import retrieve_documents, retrieve_with_year_filter
from .router import compute_rag_confidence, route_query
from .answer_formatter import format_rag_response, format_llm_fallback_response
