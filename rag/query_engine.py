"""
DEPRECATED MODULE.
Legacy routing pipeline removed.
Do not use.
Single-brain architecture enforced via langchain_orchestrator.

All queries must route through:
    rag.langchain_orchestrator.answer_query_simple()
with ENABLE_QUERY_PLANNER=true.

This file is retained as a tombstone to prevent accidental re-introduction
of the legacy dual-routing pipeline. Do not add new code here.
"""

raise ImportError(
    "query_engine is deprecated. "
    "Use rag.langchain_orchestrator.answer_query_simple() instead."
)
