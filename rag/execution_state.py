"""
Execution State — deterministic routing contract for query processing.

``ExecutionState`` formalises the decision state that emerges after retrieval
and adequacy evaluation.  It replaces loose ad-hoc ``execution_path`` string
derivation with a single ``resolve_mode()`` call that deterministically maps
evidence counts to one of four canonical modes.

Canonical modes (``final_mode``):
    ``RAG``           — RAG evidence is usable, no web evidence.
    ``RAG_PLUS_WEB``  — Both RAG and web evidence available.
    ``WEB_ONLY``      — Only web evidence available.
    ``LLM_ONLY``      — No usable evidence; LLM answers from general knowledge.

Legacy compatibility:
    Downstream consumers (``ConfidenceScorer``, ``_normalize_sources``,
    ``_derive_confidence_and_disclosure``) use substring checks such as
    ``"RAG" in execution_path`` and ``"WEB_SEARCH" in execution_path``.
    The ``execution_path`` property maps ``final_mode`` to the legacy string
    format so that **no behaviour change** occurs in existing consumers.
"""

from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Mode ↔ legacy-path mappings
# ---------------------------------------------------------------------------

_MODE_TO_PATH: Dict[str, str] = {
    "RAG_PLUS_WEB": "WEB_SEARCH+RAG",
    "WEB_ONLY":     "WEB_SEARCH",
    "RAG":          "RAG",
    "LLM_ONLY":     "LLM_ONLY",
}

_MODE_TO_ANSWER_TYPE: Dict[str, str] = {
    "RAG_PLUS_WEB": "sagealpha_hybrid_search",
    "WEB_ONLY":     "sagealpha_ai_search",
    "RAG":          "sagealpha_rag",
    "LLM_ONLY":     "sagealpha_rag",
}


class ExecutionState:
    """Deterministic routing contract resolved after retrieval + adequacy.

    Parameters
    ----------
    retrieval_count : int
        Number of **usable** RAG documents (0 when adequacy forced LLM_ONLY,
        even if raw retrieval returned documents — those are preserved for
        audit only).
    web_count : int
        Number of web-search result documents.
    adequacy_decision : str
        The adequacy evaluator's decision string (``"USE_RAG"``,
        ``"ESCALATE_WEB"``, ``"LLM_ONLY"``) or ``"N/A"`` when no adequacy
        evaluation ran.
    numeric_intent : bool
        Whether the query carries numeric / exact-value intent.
    requested_year : str or None
        Extracted fiscal year (``"FYxxxx"``) or ``None``.
    """

    def __init__(
        self,
        retrieval_count: int,
        web_count: int,
        adequacy_decision: str,
        numeric_intent: bool,
        requested_year: Optional[str],
    ):
        self.retrieval_count = retrieval_count
        self.web_count = web_count
        self.adequacy_decision = adequacy_decision
        self.numeric_intent = numeric_intent
        self.requested_year = requested_year

        self.final_mode: Optional[str] = None
        self.escalation_reason: Optional[str] = None

    # --- mode resolution ------------------------------------------------

    def resolve_mode(self) -> None:
        """Deterministically resolve ``final_mode`` from evidence counts.

        Must be called exactly once, after all retrieval and adequacy steps
        have completed and the counts are final.
        """
        if self.web_count > 0 and self.retrieval_count > 0:
            self.final_mode = "RAG_PLUS_WEB"
        elif self.web_count > 0:
            self.final_mode = "WEB_ONLY"
        elif self.retrieval_count > 0:
            self.final_mode = "RAG"
        else:
            self.final_mode = "LLM_ONLY"

    # --- legacy compatibility -------------------------------------------

    @property
    def execution_path(self) -> str:
        """Legacy-compatible execution-path string for downstream consumers.

        Maps ``final_mode`` to the string format expected by
        ``ConfidenceScorer``, ``_normalize_sources``, and
        ``_derive_confidence_and_disclosure`` (which use substring checks
        like ``"RAG" in execution_path``).
        """
        if self.final_mode is None:
            return "UNKNOWN"
        return _MODE_TO_PATH.get(self.final_mode, "UNKNOWN")

    @property
    def answer_type(self) -> str:
        """API-facing answer-type label derived from ``final_mode``."""
        if self.final_mode is None:
            return "sagealpha_rag"
        return _MODE_TO_ANSWER_TYPE.get(self.final_mode, "sagealpha_rag")

    # --- structured log payload -----------------------------------------

    def to_log_dict(self) -> Dict[str, Any]:
        """Return a dict suitable for ``logger.info(..., extra=...)``."""
        return {
            "event_type": "execution_contract",
            "retrieval_count": self.retrieval_count,
            "web_count": self.web_count,
            "adequacy_decision": self.adequacy_decision,
            "final_mode": self.final_mode,
            "numeric_intent": self.numeric_intent,
            "requested_year": self.requested_year,
        }

    def __repr__(self) -> str:
        return (
            f"ExecutionState(final_mode={self.final_mode!r}, "
            f"retrieval_count={self.retrieval_count}, "
            f"web_count={self.web_count}, "
            f"adequacy_decision={self.adequacy_decision!r})"
        )
