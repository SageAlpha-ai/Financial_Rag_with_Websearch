"""
Lightweight Observability Metrics (log-only)

Records Tier-1 answerability outcomes as structured log events.
Metrics are derived from logs by downstream log aggregation tools
(e.g. Azure Monitor, CloudWatch, ELK) — no external metric
dependencies are required.

All methods are static, never raise, and never log user content,
document text, or embeddings.
"""

import logging
from typing import Optional

from rag.query_constraints import (
    detect_numeric_intent as _detect_numeric_intent,
    extract_fiscal_year as _extract_year,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight keyword-based entity extractor for metrics tagging.
# Intentionally NOT the LLM-backed extract_company_name — metrics must
# never invoke LLM calls or add latency.
# ---------------------------------------------------------------------------

_ENTITY_MAPPINGS = {
    "oracle financial services": "Oracle Financial Services Software Ltd",
    "oracle financial": "Oracle Financial Services Software Ltd",
    "ofss": "Oracle Financial Services Software Ltd",
    "microsoft": "Microsoft",
    "apple": "Apple",
    "google": "Google",
    "amazon": "Amazon",
    "meta": "Meta",
    "tesla": "Tesla",
    "nvidia": "NVIDIA",
}


def _extract_entity(query: str) -> Optional[str]:
    """Extract company/entity from query using keyword matching."""
    query_lower = query.lower()
    for key, value in _ENTITY_MAPPINGS.items():
        if key in query_lower:
            return value
    return None


# ---------------------------------------------------------------------------
# Public metrics API
# ---------------------------------------------------------------------------


class MetricsRecorder:
    """Log-only metrics recorder for Tier-1 answerability outcomes.

    Every method is ``@staticmethod``, never raises, and logs a single
    structured ``[METRICS]`` line.  These lines are designed to be
    parsed by log aggregation pipelines.
    """

    @staticmethod
    def record_blocked_query(
        question: str,
        reason: Optional[str],
        has_documents: bool,
    ) -> None:
        """Record when the answerability gate blocks an answer."""
        try:
            logger.info(
                "[METRICS] event_type=blocked_query reason=%s "
                "has_documents=%s has_numeric_intent=%s "
                "requested_entity=%s requested_year=%s",
                reason,
                has_documents,
                _detect_numeric_intent(question),
                _extract_entity(question),
                _extract_year(question),
            )
        except Exception:
            pass  # Metrics must never raise

    @staticmethod
    def record_llm_fallback(
        question: str,
        reason: Optional[str] = None,
    ) -> None:
        """Record when the LLM-only fallback path is taken."""
        try:
            logger.info(
                "[METRICS] event_type=llm_fallback reason=%s "
                "has_documents=False has_numeric_intent=%s "
                "requested_entity=%s requested_year=%s",
                reason,
                _detect_numeric_intent(question),
                _extract_entity(question),
                _extract_year(question),
            )
        except Exception:
            pass

    @staticmethod
    def record_insufficient_evidence(
        question: str,
        reason: Optional[str] = None,
    ) -> None:
        """Record when report generation is blocked for lack of evidence."""
        try:
            logger.info(
                "[METRICS] event_type=insufficient_evidence reason=%s "
                "has_documents=False has_numeric_intent=%s "
                "requested_entity=%s requested_year=%s",
                reason,
                _detect_numeric_intent(question),
                _extract_entity(question),
                _extract_year(question),
            )
        except Exception:
            pass

    @staticmethod
    def record_rag_answer(
        question: str,
        has_documents: bool,
    ) -> None:
        """Record when a RAG-grounded answer is successfully returned."""
        try:
            logger.info(
                "[METRICS] event_type=rag_answer reason=None "
                "has_documents=%s has_numeric_intent=%s "
                "requested_entity=%s requested_year=%s",
                has_documents,
                _detect_numeric_intent(question),
                _extract_entity(question),
                _extract_year(question),
            )
        except Exception:
            pass
