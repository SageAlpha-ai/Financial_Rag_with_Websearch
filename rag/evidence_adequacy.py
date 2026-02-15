"""
Evidence Adequacy Evaluator — **DEPRECATED**.

.. deprecated:: 2026-02
    Replaced by ``rag.answerability_model.AnswerabilityModel``, a
    feature-based learned routing classifier that uses linear weights +
    softmax instead of hardcoded thresholds.

    This module is retained for backward compatibility and telemetry
    reference only.  **No active code path calls
    ``EvidenceAdequacyEvaluator.evaluate()`` any more.**

    New code should use::

        from rag.answerability_model import AnswerabilityFeatures, AnswerabilityModel
        result = AnswerabilityModel.predict(features)

Legacy description
------------------
Computed a composite score that decided whether retrieved RAG documents
were sufficient to answer a query, or whether the system should escalate
to web search or fall back to LLM-only.

Scoring components (weights from config):
    - **llm_score**       — LLM-judged relevance of the top documents.
    - **retrieval_score** — cross-encoder relevance quality of the best
      match (higher is better, normalised to [0, 1]).
    - **doc_type_score**  — metadata heuristic for document-type fit.

Decision thresholds (from config):
    - >= use_rag threshold   → USE_RAG
    - >= escalate_web threshold → ESCALATE_WEB
    - below both            → LLM_ONLY

Configuration is read from ``config.settings.get_config().evidence_adequacy``.
If the config object is missing or malformed, ``DEFAULT_ADEQUACY_CONFIG``
is used as a safety fallback and a single WARNING is logged.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from langchain_openai import AzureChatOpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fallback defaults — used ONLY if config/settings.py is missing or
# malformed.  These values must stay in sync with
# config.settings.EVIDENCE_ADEQUACY_CONFIG.  They exist solely as a
# safety net; in production the config object is the single source of
# truth and these values are never consulted.
# ---------------------------------------------------------------------------

DEFAULT_ADEQUACY_CONFIG: Dict[str, Any] = {
    "weights": {
        "llm": 0.35,
        "retrieval": 0.45,
        "doc_type": 0.20,
    },
    "thresholds": {
        "use_rag": 0.60,
        "escalate_web": 0.40,
    },
    "retrieval_distance_max": 1.5,
    "retrieval_weights": {
        "base_similarity": 0.4,
        "gap": 0.2,
        "variance": 0.15,
        "diversity": 0.15,
        "alignment": 0.10,
    },
    "metadata_alignment": {
        "enabled": True,
        "company_boost_weight": 0.08,
        "year_boost_weight": 0.05,
        "max_total_boost": 0.15,
    },
    "retrieval_stability": {
        "enabled": True,
        "dominance_threshold": 0.7,
        "dominance_penalty": 0.02,
        "boost_dampening_threshold": 0.8,
    },
    "retrieval_trust": {
        "enabled": True,
        "distance_gap_threshold": 0.25,
        "min_entropy_normalizer": 1.0,
    },
    "version": "v1_hybrid",
}

# Guard: emit the fallback WARNING at most once per process.
_fallback_warning_emitted: bool = False

# ---------------------------------------------------------------------------
# Numeric-intent detection (lightweight, no external dependency)
# ---------------------------------------------------------------------------

_NUMERIC_PATTERNS: List[str] = [
    r"\d+",
    r"%",
    r"\$",
    r"\btotal\b",
    r"\bexact\b",
    r"\brate\b",
    r"\bvalue\b",
    r"\brevenue\b",
    r"\bamount\b",
    r"\bnumber\b",
    r"\bcount\b",
]

_FINANCIAL_DOC_TYPES: set[str] = {
    "annual_report",
    "financial_statement",
    "10-k",
    "10-q",
    "earnings_release",
    "quarterly_report",
    "investor_presentation",
}


def _has_numeric_intent(query: str) -> bool:
    """Return ``True`` when >=2 numeric-intent signals are present."""
    query_lower = query.lower()
    hits = sum(
        1
        for pattern in _NUMERIC_PATTERNS
        if re.search(pattern, query_lower)
    )
    return hits >= 2


# ---------------------------------------------------------------------------
# Config resolution (with structural validation)
# ---------------------------------------------------------------------------


def _resolve_adequacy_config() -> Dict[str, Any]:
    """Load and validate adequacy config from ``config.settings``.

    Returns the config dict from ``get_config().evidence_adequacy`` when it
    exists and passes structural validation.  Falls back to
    ``DEFAULT_ADEQUACY_CONFIG`` otherwise, emitting a single WARNING per
    process to avoid log spam.

    No circular imports: ``config.settings`` does not import from ``rag.*``.
    """
    global _fallback_warning_emitted

    # --- Attempt to load from config object ---
    adequacy_cfg = None
    try:
        from config.settings import get_config
        cfg = get_config()
        adequacy_cfg = getattr(cfg, "evidence_adequacy", None)
    except Exception:
        # Config system itself is broken — fall through to fallback.
        adequacy_cfg = None

    if adequacy_cfg is None:
        if not _fallback_warning_emitted:
            logger.warning(
                "[ADEQUACY] evidence_adequacy config missing from settings "
                "— using DEFAULT_ADEQUACY_CONFIG as safety fallback"
            )
            _fallback_warning_emitted = True
        return DEFAULT_ADEQUACY_CONFIG

    # --- Structural validation ---
    try:
        w = adequacy_cfg["weights"]
        if not (
            isinstance(w, dict)
            and "llm" in w
            and "retrieval" in w
            and "doc_type" in w
        ):
            raise KeyError("weights incomplete")

        t = adequacy_cfg["thresholds"]
        if not (
            isinstance(t, dict)
            and "use_rag" in t
            and "escalate_web" in t
        ):
            raise KeyError("thresholds incomplete")

        if "retrieval_distance_max" not in adequacy_cfg:
            raise KeyError("retrieval_distance_max missing")

    except (KeyError, TypeError) as exc:
        if not _fallback_warning_emitted:
            logger.warning(
                "[ADEQUACY] evidence_adequacy config malformed (%s) "
                "— using DEFAULT_ADEQUACY_CONFIG as safety fallback",
                exc,
            )
            _fallback_warning_emitted = True
        return DEFAULT_ADEQUACY_CONFIG

    return adequacy_cfg


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class EvidenceAdequacyEvaluator:
    """**DEPRECATED** — replaced by ``rag.answerability_model.AnswerabilityModel``.

    Retained for backward compatibility.  No active orchestrator code path
    calls ``evaluate()`` any more.  The learned classifier in
    ``AnswerabilityModel.predict()`` replaces all threshold-based routing.
    """

    @staticmethod
    def evaluate(
        question: str,
        rag_docs: List[str],
        rag_metas: List[Dict],
        best_distance: float,
        llm: AzureChatOpenAI,
        retrieval_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Score evidence adequacy and return a routing decision.

        All weights, thresholds, and distance normalisation parameters are
        read from ``config.settings.EVIDENCE_ADEQUACY_CONFIG`` (via
        ``get_config().evidence_adequacy``).  Hardcoded defaults are used
        only if that config is missing or malformed.

        **Semantic direction**: the primary retrieval signal is now
        ``best_cross_score`` (higher = stronger evidence, normalised
        [0, 1]).  All comparison logic uses *higher-is-better* semantics.
        The legacy ``best_distance`` parameter is kept for backward
        compatibility and telemetry only; it is **never** used as the
        primary scoring signal when a cross-encoder score is present.

        Args:
            question: The user query.
            rag_docs: Retrieved document texts (may be empty).
            rag_metas: Corresponding metadata dicts (parallel to *rag_docs*).
            best_distance: **Deprecated positional arg.** Callers now pass
                ``retrieval_metrics["best_cross_score"]`` here.  Retained
                only for API compatibility; ignored when
                ``retrieval_metrics["best_cross_score"]`` is available.
            llm: An initialised ``AzureChatOpenAI`` instance used for the
                LLM-judged adequacy check (invoked with temperature 0).

        Returns:
            A dict with exactly five keys::

                {
                    "llm_score":        float,   # 0–1
                    "retrieval_score":  float,   # 0–1
                    "doc_type_score":   float,   # 0–1
                    "final_score":      float,   # 0–1
                    "decision":         str,     # USE_RAG | ESCALATE_WEB | LLM_ONLY
                }
        """
        cfg = _resolve_adequacy_config()
        weights = cfg["weights"]
        thresholds = cfg["thresholds"]
        max_dist = float(cfg.get("retrieval_distance_max", 1.5))
        rw = cfg.get("retrieval_weights") or DEFAULT_ADEQUACY_CONFIG.get("retrieval_weights", {})
        w_base = float(rw.get("base_similarity", 0.4))
        w_gap = float(rw.get("gap", 0.2))
        w_var = float(rw.get("variance", 0.15))
        w_div = float(rw.get("diversity", 0.15))
        w_align = float(rw.get("alignment", 0.10))

        # ------------------------------------------------------------------
        # 1. Retrieval score (multi-signal when retrieval_metrics provided)
        # ------------------------------------------------------------------
        rm = retrieval_metrics or {}

        # --- Primary signal: cross-encoder score (higher is better) ---
        # The normalised cross-encoder score is the authoritative
        # retrieval quality signal.  It is always in [0, 1] after
        # sigmoid normalization in cross_encoder_reranker.py.
        best_cross_score = float(rm.get("best_cross_score", 0.0))

        # --- Legacy signal: cosine distance (lower is better) ---
        # Preserved in metrics for telemetry / dashboards only.
        # NEVER used as the base_similarity when cross-encoder is active.
        cosine_best_distance = float(rm.get("best_distance", 2.0))

        distance_gap = float(rm.get("distance_gap", 0.0))
        distance_variance = float(rm.get("distance_variance", 0.0))
        diversity_ratio = float(rm.get("diversity_ratio", 1.0))
        year_match_ratio = float(rm.get("year_match_ratio", 1.0))
        company_match_ratio = float(rm.get("company_match_ratio", 1.0))

        # --- Determine base_similarity (always higher-is-better [0, 1]) ---
        _cross_encoder_active = best_cross_score > 0.0

        if _cross_encoder_active:
            # Cross-encoder: already normalised [0, 1], higher = better
            base_similarity = max(0.0, min(1.0, best_cross_score))
            logger.debug(
                "[ADEQUACY] Using cross-encoder score as base_similarity: %.4f "
                "(higher is better)",
                base_similarity,
            )
        else:
            # Legacy fallback: convert cosine distance → similarity
            max_dist_safe = max(1e-6, max_dist)
            base_similarity = max(
                0.0, min(1.0, 1.0 - cosine_best_distance / max_dist_safe)
            )
            logger.debug(
                "[ADEQUACY] No cross-encoder — using cosine distance→similarity "
                "as base_similarity: %.4f (distance=%.4f)",
                base_similarity,
                cosine_best_distance,
            )

        # --- Auxiliary signals (unchanged — these are already [0, 1]) ---
        gap_score = min(1.0, distance_gap / 0.15) if distance_gap >= 0 else 0.0
        variance_penalty = max(0.0, 1.0 - distance_variance * 5)
        diversity_bonus = max(0.0, min(1.0, diversity_ratio))
        alignment_bonus = (year_match_ratio + company_match_ratio) / 2.0
        alignment_bonus = max(0.0, min(1.0, alignment_bonus))

        retrieval_score = (
            w_base * base_similarity
            + w_gap * gap_score
            + w_var * variance_penalty
            + w_div * diversity_bonus
            + w_align * alignment_bonus
        )
        retrieval_score = max(0.0, min(1.0, retrieval_score))
        trust_score = float(rm.get("trust_score", 1.0))
        retrieval_score = retrieval_score * trust_score
        retrieval_score = max(0.0, min(1.0, retrieval_score))

        logger.debug(
            "retrieval_signal_computed",
            extra={
                "event_type": "retrieval_signal",
                "cross_encoder_active": _cross_encoder_active,
                "best_cross_score": best_cross_score,
                "cosine_best_distance": cosine_best_distance,
                "base_similarity": base_similarity,
                "distance_gap": distance_gap,
                "distance_variance": distance_variance,
                "diversity_ratio": diversity_ratio,
                "year_match_ratio": year_match_ratio,
                "company_match_ratio": company_match_ratio,
                "retrieval_score": retrieval_score,
            },
        )

        # ------------------------------------------------------------------
        # 1b. Deterministic retrieval dominance (higher-is-better)
        # ------------------------------------------------------------------
        # When retrieval quality is objectively strong AND enough documents
        # are present, bypass subjective scoring to guarantee USE_RAG.
        #
        # Cross-encoder path: best_cross_score >= 0.85 (higher is better)
        # Legacy distance path: cosine_best_distance < 0.25 (lower is better)
        _dominance_by_cross = (
            _cross_encoder_active
            and best_cross_score >= 0.85
            and len(rag_docs) >= 3
        )
        _dominance_by_dist = (
            not _cross_encoder_active
            and cosine_best_distance < 0.25
            and len(rag_docs) >= 3
        )
        if _dominance_by_cross or _dominance_by_dist:
            logger.debug(
                "[ADEQUACY] retrieval dominance short-circuit → USE_RAG "
                "(cross_encoder_active=%s, best_cross_score=%.4f, "
                "cosine_best_distance=%.4f, doc_count=%d)",
                _cross_encoder_active,
                best_cross_score,
                cosine_best_distance,
                len(rag_docs),
            )
            # Still compute LLM/doc-type for telemetry completeness
            llm_score = _llm_adequacy_score(question, rag_docs, llm)
            doc_type_score = _doc_type_score(question, rag_metas)
            return {
                "llm_score": round(llm_score, 4),
                "retrieval_score": round(retrieval_score, 4),
                "doc_type_score": round(doc_type_score, 4),
                "final_score": round(retrieval_score, 4),
                "decision": "USE_RAG",
            }

        # ------------------------------------------------------------------
        # 2. LLM-judged adequacy score
        # ------------------------------------------------------------------
        llm_score = _llm_adequacy_score(question, rag_docs, llm)
        logger.debug("[ADEQUACY] llm_score=%.4f", llm_score)

        # ------------------------------------------------------------------
        # 3. Document-type heuristic score
        # ------------------------------------------------------------------
        doc_type_score = _doc_type_score(question, rag_metas)
        logger.debug("[ADEQUACY] doc_type_score=%.4f", doc_type_score)

        # ------------------------------------------------------------------
        # 4a. Strong retrieval short-circuit
        # ------------------------------------------------------------------
        # When retrieval quality alone is high enough, bypass composite
        # scoring entirely.  This prevents a mediocre LLM adequacy score
        # from dragging strong internal evidence into ESCALATE_WEB.
        if retrieval_score >= 0.75:
            logger.debug("[ADEQUACY] strong retrieval short-circuit → USE_RAG")
            return {
                "llm_score": round(llm_score, 4),
                "retrieval_score": round(retrieval_score, 4),
                "doc_type_score": round(doc_type_score, 4),
                "final_score": round(retrieval_score, 4),
                "decision": "USE_RAG",
            }

        # ------------------------------------------------------------------
        # 4. Composite score
        # ------------------------------------------------------------------
        # Dynamic weight override for numeric-intent queries: retrieval
        # signal is boosted so that concrete financial evidence is not
        # drowned out by a subjective LLM adequacy score.
        numeric_intent = _has_numeric_intent(question)
        if numeric_intent:
            eff_w_llm = 0.25
            eff_w_ret = 0.55
            eff_w_doc = 0.20
            logger.debug(
                "[ADEQUACY] numeric-intent weight override: "
                "llm=%.2f retrieval=%.2f doc_type=%.2f",
                eff_w_llm, eff_w_ret, eff_w_doc,
            )
        else:
            eff_w_llm = float(weights["llm"])
            eff_w_ret = float(weights["retrieval"])
            eff_w_doc = float(weights["doc_type"])

        final_score = (
            eff_w_llm * llm_score
            + eff_w_ret * retrieval_score
            + eff_w_doc * doc_type_score
        )
        final_score = round(final_score, 4)

        # ------------------------------------------------------------------
        # 5. Decision
        # ------------------------------------------------------------------
        if final_score >= thresholds["use_rag"]:
            decision = "USE_RAG"
        elif final_score >= thresholds["escalate_web"]:
            decision = "ESCALATE_WEB"
        else:
            decision = "LLM_ONLY"

        # ------------------------------------------------------------------
        # 5b. Retrieval-score floor: LLM adequacy cannot drag a strong
        #     retrieval below USE_RAG.
        # ------------------------------------------------------------------
        if retrieval_score >= 0.7 and decision != "USE_RAG":
            logger.debug(
                "[ADEQUACY] retrieval floor override: %s → USE_RAG "
                "(retrieval_score=%.4f, composite=%.4f)",
                decision,
                retrieval_score,
                final_score,
            )
            decision = "USE_RAG"

        logger.debug(
            "[ADEQUACY] final_score=%.4f → decision=%s",
            final_score,
            decision,
        )

        return {
            "llm_score": round(llm_score, 4),
            "retrieval_score": round(retrieval_score, 4),
            "doc_type_score": round(doc_type_score, 4),
            "final_score": final_score,
            "decision": decision,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _llm_adequacy_score(
    question: str,
    rag_docs: List[str],
    llm: AzureChatOpenAI,
) -> float:
    """Ask the LLM to judge how well the documents answer the question.

    Returns a float in [0, 1].  Defaults to 0.5 on any failure.
    """
    if not rag_docs:
        return 0.0

    # Truncate to first 3 docs, total <= 1500 chars
    truncated_parts: List[str] = []
    chars_remaining = 1500
    for doc in rag_docs[:3]:
        if chars_remaining <= 0:
            break
        chunk = doc[:chars_remaining]
        truncated_parts.append(chunk)
        chars_remaining -= len(chunk)

    docs_text = "\n---\n".join(truncated_parts)

    prompt = (
        "You are evaluating document adequacy.\n"
        f"Question: {question}\n"
        f"Documents:\n{docs_text}\n"
        'Return JSON: {"adequacy": 0-1 score}'
    )

    try:
        response = llm.invoke(prompt, temperature=0)
        usage = None
        try:
            meta = getattr(response, "response_metadata", None) or {}
            usage = meta.get("usage") or meta.get("token_usage")
        except Exception:
            pass
        if usage and isinstance(usage, dict):
            try:
                from rag.telemetry import cost_tracker
                from config.settings import get_config
                model = getattr(get_config().azure_openai, "large_chat_deployment", "gpt-4o-mini")
                cost_tracker.record(
                    stage="adequacy_llm",
                    model=model,
                    input_tokens=int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
                    output_tokens=int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
                )
            except Exception:
                pass
        raw = response.content if hasattr(response, "content") else str(response)
        match = re.search(r"\{[^}]*\}", raw)
        if match:
            parsed = json.loads(match.group())
            score = float(parsed.get("adequacy", 0.5))
            return max(0.0, min(1.0, score))
    except Exception as exc:
        logger.debug("[ADEQUACY] LLM adequacy check failed: %s", exc)

    return 0.5


def _doc_type_score(question: str, rag_metas: List[Dict]) -> float:
    """Heuristic: boost when doc types match numeric-intent queries."""
    if not rag_metas:
        return 0.5

    has_financial_type = any(
        str(meta.get("type", "")).lower() in _FINANCIAL_DOC_TYPES
        for meta in rag_metas
    )

    if has_financial_type and _has_numeric_intent(question):
        return 0.8

    return 0.5
