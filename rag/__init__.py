"""RAG module - Single-brain architecture via langchain_orchestrator.

All query routing is handled by langchain_orchestrator.answer_query_simple()
with ENABLE_QUERY_PLANNER=true.  Legacy exports (retrieve_with_year_filter,
route_query, format_rag_response, format_llm_fallback_response) have been
removed.  Individual sub-modules (retriever, router, answer_formatter) remain
available for internal use by the orchestrator.
"""
from .retriever import retrieve_documents
from .router import compute_rag_confidence
