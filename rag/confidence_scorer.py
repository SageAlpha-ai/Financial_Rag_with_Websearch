"""
Confidence Scoring for RAG Answers.

Produces a deterministic 0-100 numeric score and a categorical confidence
level (HIGH / MEDIUM / LOW) based on the execution path and the volume of
supporting evidence.

Scoring breakdown:
  - Base score derived from execution path (RAG > WEB > LLM_ONLY).
  - Evidence-depth bonus scaled by the number of source documents.
  - Hybrid bonus when both RAG and web evidence are present.

Level thresholds are aligned with ``_derive_confidence_and_disclosure``
in ``langchain_orchestrator.py`` so that the numeric score and the
path-based disclosure note never contradict each other.
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class ConfidenceScorer:
    """Deterministic confidence scorer based on execution path and evidence."""

    @staticmethod
    def calculate_score(
        execution_path: str,
        rag_metas: List[Dict],
        web_docs: List[Dict],
    ) -> float:
        """Return a 0-100 confidence score.

        Args:
            execution_path: The resolved execution path string
                (e.g. ``"RAG+LLM"``, ``"WEB_SEARCH+LLM"``, ``"LLM_ONLY"``).
            rag_metas: Metadata dicts for retrieved RAG documents.
            web_docs: Web-search result dicts.

        Returns:
            A ``float`` in ``[0, 100]``.
        """
        has_rag = bool(rag_metas)
        has_web = bool(web_docs)

        # --- SYSTEM path (deterministic capability response) ---
        if execution_path == "SYSTEM":
            return 95.0

        score = 0.0

        # --- Base score from execution path ---
        if "RAG" in execution_path and has_rag:
            score += 60.0
        elif "WEB_SEARCH" in execution_path and has_web:
            score += 40.0
        else:
            # LLM_ONLY or unrecognised path
            score += 15.0

        # --- Evidence-depth bonus (up to 25) ---
        doc_count = len(rag_metas) + len(web_docs)
        score += min(25.0, doc_count * 5.0)

        # --- Hybrid bonus (both RAG and web present, up to 15) ---
        if has_rag and has_web:
            score += 15.0

        return min(100.0, max(0.0, score))

    @staticmethod
    def get_confidence_level(score: float) -> str:
        """Map a numeric score to a categorical confidence level.

        Thresholds are aligned with ``_derive_confidence_and_disclosure``:

        ============  =======================================
        Level         Condition
        ============  =======================================
        ``HIGH``      score >= 75  (RAG-backed answers)
        ``MEDIUM``    40 <= score < 75  (web-only answers)
        ``LOW``       score < 40  (LLM-only / no evidence)
        ============  =======================================
        """
        if score >= 75.0:
            return "HIGH"
        if score >= 40.0:
            return "MEDIUM"
        return "LOW"
