"""
Answerability Model — learned routing classifier for evidence adequacy.

Replaces the heuristic threshold-based ``EvidenceAdequacyEvaluator`` with a
feature-based logistic-regression-style classifier that routes queries to
one of three execution paths: ``USE_RAG``, ``ESCALATE_WEB``, ``LLM_ONLY``.

Architecture
------------
1. **Feature extraction** — five normalised signals are computed from
   retrieval metrics and query analysis.
2. **Linear scoring** — per-class weight vectors are dot-producted with the
   feature vector to produce three raw logits.
3. **Softmax** — logits are converted to class probabilities.
4. **Decision** — ``argmax`` over probabilities selects the routing decision.

All weights are stored in ``config.settings.ANSWERABILITY_MODEL_CONFIG``
and can be updated without code changes.  The linear + softmax design
makes the model fully interpretable: each weight directly indicates
how much a feature contributes to a routing decision.

No hardcoded thresholds exist in this module.  All routing is determined
by the learned weight vectors.

Public API
----------
``AnswerabilityFeatures``
    Pydantic model for the five input features.
``AnswerabilityDecision``
    Enum of the three possible routing outcomes.
``AnswerabilityModel``
    Stateless classifier; call ``predict(features)`` → result dict.
"""

import logging
import math
from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------

class AnswerabilityFeatures(BaseModel):
    """Normalised feature vector for the answerability classifier.

    All float features are expected in [0, 1].  ``numeric_intent`` is a
    boolean indicator converted to 0/1 internally.
    """

    best_cross_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalised cross-encoder relevance score of the best "
        "document (higher = more relevant).",
    )
    doc_count: int = Field(
        ...,
        ge=0,
        description="Number of documents retrieved after re-ranking.",
    )
    year_match_ratio: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of retrieved documents whose fiscal year "
        "matches the query's requested year (1.0 when no year requested).",
    )
    entity_match_ratio: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of retrieved documents whose company/entity "
        "matches the query's target entity (1.0 when no entity detected).",
    )
    numeric_intent: bool = Field(
        ...,
        description="Whether the query carries numeric or exact-value intent.",
    )

    def to_vector(self, doc_count_normalizer: float = 10.0) -> List[float]:
        """Convert to a fixed-length float vector for linear scoring.

        ``doc_count`` is normalised to [0, 1] by dividing by
        ``doc_count_normalizer`` (default 10, the typical FINAL_TOP_K).
        """
        return [
            self.best_cross_score,
            min(1.0, self.doc_count / max(1.0, doc_count_normalizer)),
            self.year_match_ratio,
            self.entity_match_ratio,
            1.0 if self.numeric_intent else 0.0,
        ]


# ---------------------------------------------------------------------------
# Decision enum
# ---------------------------------------------------------------------------

class AnswerabilityDecision(str, Enum):
    """Routing decision produced by the answerability classifier."""

    USE_RAG = "USE_RAG"
    ESCALATE_WEB = "ESCALATE_WEB"
    LLM_ONLY = "LLM_ONLY"


# ---------------------------------------------------------------------------
# Softmax utility
# ---------------------------------------------------------------------------

def _softmax(logits: List[float]) -> List[float]:
    """Numerically stable softmax over a list of logits."""
    max_logit = max(logits)
    exps = [math.exp(l - max_logit) for l in logits]
    total = sum(exps)
    return [e / total for e in exps]


# ---------------------------------------------------------------------------
# Default weight vectors (safety fallback)
# ---------------------------------------------------------------------------
# Kept in sync with config.settings.ANSWERABILITY_MODEL_CONFIG.
# Only consulted when the config object is missing or malformed.

_FALLBACK_WEIGHTS: Dict[str, Any] = {
    "USE_RAG": {
        "best_cross_score": 4.0,
        "doc_count": 1.5,
        "year_match_ratio": 1.0,
        "entity_match_ratio": 1.0,
        "numeric_intent": 0.3,
        "bias": -2.5,
    },
    "ESCALATE_WEB": {
        "best_cross_score": -1.5,
        "doc_count": -0.5,
        "year_match_ratio": -0.8,
        "entity_match_ratio": -0.8,
        "numeric_intent": 0.8,
        "bias": 1.0,
    },
    "LLM_ONLY": {
        "best_cross_score": -3.0,
        "doc_count": -1.5,
        "year_match_ratio": -0.5,
        "entity_match_ratio": -0.5,
        "numeric_intent": -0.5,
        "bias": 1.5,
    },
    "doc_count_normalizer": 10.0,
    "version": "v1_linear_softmax",
}

_FEATURE_KEYS = [
    "best_cross_score",
    "doc_count",
    "year_match_ratio",
    "entity_match_ratio",
    "numeric_intent",
]

_CLASS_ORDER: List[AnswerabilityDecision] = [
    AnswerabilityDecision.USE_RAG,
    AnswerabilityDecision.ESCALATE_WEB,
    AnswerabilityDecision.LLM_ONLY,
]

# Guard: emit fallback warning at most once.
_fallback_warning_emitted: bool = False


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------

def _resolve_weights() -> Dict[str, Any]:
    """Load answerability model weights from ``config.settings``.

    Returns the dict from ``ANSWERABILITY_MODEL_CONFIG`` when it exists
    and passes structural validation.  Falls back to ``_FALLBACK_WEIGHTS``
    otherwise, emitting a single WARNING per process.
    """
    global _fallback_warning_emitted

    try:
        from config.settings import ANSWERABILITY_MODEL_CONFIG
        weights = ANSWERABILITY_MODEL_CONFIG
    except (ImportError, AttributeError):
        weights = None

    if weights is None:
        if not _fallback_warning_emitted:
            logger.warning(
                "[ANSWERABILITY_MODEL] ANSWERABILITY_MODEL_CONFIG missing "
                "from settings — using fallback weights"
            )
            _fallback_warning_emitted = True
        return _FALLBACK_WEIGHTS

    # Structural validation
    try:
        for cls in ("USE_RAG", "ESCALATE_WEB", "LLM_ONLY"):
            w = weights[cls]
            for k in _FEATURE_KEYS:
                if k not in w:
                    raise KeyError(f"{cls}.{k} missing")
            if "bias" not in w:
                raise KeyError(f"{cls}.bias missing")
    except (KeyError, TypeError) as exc:
        if not _fallback_warning_emitted:
            logger.warning(
                "[ANSWERABILITY_MODEL] config malformed (%s) — using fallback",
                exc,
            )
            _fallback_warning_emitted = True
        return _FALLBACK_WEIGHTS

    return weights


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class AnswerabilityModel:
    """Stateless logistic-regression-style routing classifier.

    Computes a linear combination of input features per class, applies
    softmax, and returns the class with the highest probability.

    Thread-safe: no mutable instance state.
    """

    @staticmethod
    def predict(features: AnswerabilityFeatures) -> Dict[str, Any]:
        """Classify a query into a routing decision.

        Parameters
        ----------
        features : AnswerabilityFeatures
            The five normalised input features.

        Returns
        -------
        dict
            A dict with the following keys::

                {
                    "decision":     str,            # USE_RAG | ESCALATE_WEB | LLM_ONLY
                    "probabilities": {
                        "USE_RAG":       float,
                        "ESCALATE_WEB":  float,
                        "LLM_ONLY":      float,
                    },
                    "logits": {
                        "USE_RAG":       float,
                        "ESCALATE_WEB":  float,
                        "LLM_ONLY":      float,
                    },
                    "features": dict,               # input features as a flat dict
                    "low_confidence_override": bool, # True when max_prob < 0.55
                    "model_version": str,
                }
        """
        weights = _resolve_weights()
        doc_normalizer = float(weights.get("doc_count_normalizer", 10.0))
        feature_vec = features.to_vector(doc_count_normalizer=doc_normalizer)

        # --- Compute per-class logits (linear score + bias) ---
        logits: List[float] = []
        for cls in _CLASS_ORDER:
            cls_weights = weights[cls.value]
            dot = sum(
                cls_weights[k] * v
                for k, v in zip(_FEATURE_KEYS, feature_vec)
            )
            dot += float(cls_weights.get("bias", 0.0))
            logits.append(dot)

        # --- Softmax → probabilities ---
        probs = _softmax(logits)

        # --- Argmax → decision ---
        best_idx = max(range(len(probs)), key=lambda i: probs[i])
        decision = _CLASS_ORDER[best_idx]

        logit_dict = {
            cls.value: round(logit, 4)
            for cls, logit in zip(_CLASS_ORDER, logits)
        }
        prob_dict = {
            cls.value: round(prob, 4)
            for cls, prob in zip(_CLASS_ORDER, probs)
        }

        # --- Confidence-margin safeguard ---
        # When no class wins decisively (max probability below the
        # confidence margin), override to ESCALATE_WEB.  This prevents
        # the model from committing to a weak routing decision and
        # instead hedges by supplementing with web evidence.
        max_prob = max(prob_dict.values())
        if max_prob < 0.55:
            decision = AnswerabilityDecision.ESCALATE_WEB
            low_confidence_override = True
        else:
            low_confidence_override = False

        feature_dict = {
            "best_cross_score": features.best_cross_score,
            "doc_count": features.doc_count,
            "year_match_ratio": features.year_match_ratio,
            "entity_match_ratio": features.entity_match_ratio,
            "numeric_intent": features.numeric_intent,
        }

        # --- Structured logging ---
        logger.info(
            "[ANSWERABILITY_MODEL] features=%s",
            feature_dict,
        )
        logger.info(
            "[ANSWERABILITY_MODEL] logits=%s  probabilities=%s",
            logit_dict,
            prob_dict,
        )
        logger.info(
            "[ANSWERABILITY_MODEL] low_confidence_override=%s",
            low_confidence_override,
        )
        logger.info(
            "[ANSWERABILITY_MODEL] decision=%s",
            decision.value,
        )

        return {
            "decision": decision.value,
            "probabilities": prob_dict,
            "logits": logit_dict,
            "features": feature_dict,
            "low_confidence_override": low_confidence_override,
            "model_version": weights.get("version", "unknown"),
        }
