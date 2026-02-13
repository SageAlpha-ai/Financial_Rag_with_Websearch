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
import re
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Query constraint extractors (duplicated from orchestrator to avoid
# circular imports — these are trivial, deterministic helpers).
# ---------------------------------------------------------------------------

_FINANCIAL_KEYWORDS = [
    "revenue", "profit", "income", "earnings", "pat",
    "ebitda", "assets", "equity", "turnover", "sales",
    "quarterly", "quarter", "q1", "q2", "q3", "q4",
    "balance sheet", "income statement", "cash flow",
    "financial statement", "annual report",
]

_YEAR_PATTERNS = [
    r'FY\s*(\d{4})',
    r'fiscal\s+year\s+(\d{4})',
    r'\b(20\d{2})\b',
    r'\b(19\d{2})\b',
]

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


def _detect_numeric_intent(query: str) -> bool:
    """Return True if the query has numeric or financial intent."""
    query_lower = query.lower()
    return any(kw in query_lower for kw in _FINANCIAL_KEYWORDS)


def _extract_year(query: str) -> Optional[str]:
    """Extract fiscal year from query (FYxxxx or None)."""
    for pattern in _YEAR_PATTERNS:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return f"FY{match.group(1)}"
    return None


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
