"""
Cross-Encoder Re-Ranker — enterprise retrieval layer.

Sits between bi-encoder retrieval (vector similarity) and evidence
adequacy scoring.  Takes the broader candidate set produced by cosine
search and re-scores every (query, document) pair through a cross-encoder
model, yielding a much more precise relevance ranking.

Design notes
------------
* **Deterministic**: temperature=0 when an LLM-based cross-encoder is used.
* **Batch scoring**: all candidates are scored in a single call.
* **Metadata preservation**: original cosine distance is kept; a new
  ``cross_score`` field is injected into each document's metadata dict.

Public API
----------
``rerank_documents(query, documents, metadatas)``
    → ``List[Tuple[str, Dict, float]]``  sorted by *cross_score* descending.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid: 1 / (1 + exp(-x))."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    # For large negative x, avoid overflow in exp(-x)
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _needs_normalization(scores: List[float]) -> bool:
    """Return True when any score falls outside [0, 1].

    Cross-encoder models that output logits (e.g. sentence-transformers
    CrossEncoder, Cohere v1) may return values in (-inf, +inf).
    Models that already output probabilities (e.g. Cohere v2 Rerank)
    will have all scores in [0, 1] and need no transformation.
    """
    return any(s < 0.0 or s > 1.0 for s in scores)


# ---------------------------------------------------------------------------
# Cross-encoder model interface
# ---------------------------------------------------------------------------

class CrossEncoderModel(Protocol):
    """Protocol that any concrete cross-encoder backend must satisfy.

    Implementations include (but are not limited to):
    * ``sentence-transformers`` ``CrossEncoder``
    * LLM-based re-rankers (Azure OpenAI, Cohere Rerank, etc.)
    * Custom ONNX / TensorRT models

    The single required method receives a batch of (query, passage) pairs
    and returns one float relevance score per pair.
    """

    def predict(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Score (query, passage) pairs.

        Parameters
        ----------
        pairs : list[tuple[str, str]]
            Each element is ``(query_text, passage_text)``.

        Returns
        -------
        list[float]
            One relevance score per input pair.  Higher is more relevant.
        """
        ...


# ---------------------------------------------------------------------------
# Default cross-encoder (configurable singleton)
# ---------------------------------------------------------------------------

_cross_encoder: Optional[CrossEncoderModel] = None


def set_cross_encoder(model: CrossEncoderModel) -> None:
    """Register a cross-encoder model for the re-ranking layer.

    Must be called once during application startup (e.g. in ``app.py``
    or a dependency-injection bootstrap).  If not called, ``rerank_documents``
    will fall back to pass-through mode (no re-ranking).
    """
    global _cross_encoder
    _cross_encoder = model
    logger.info("[RE_RANKER] Cross-encoder model registered: %s", type(model).__name__)


def get_cross_encoder() -> Optional[CrossEncoderModel]:
    """Return the currently registered cross-encoder, or ``None``."""
    return _cross_encoder


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rerank_documents(
    query: str,
    documents: List[str],
    metadatas: List[Dict[str, Any]],
) -> List[Tuple[str, Dict[str, Any], float]]:
    """Re-rank retrieved documents using the cross-encoder model.

    Parameters
    ----------
    query : str
        The user query.
    documents : list[str]
        Retrieved document texts (parallel to *metadatas*).
    metadatas : list[dict]
        Per-document metadata dicts (parallel to *documents*).

    Returns
    -------
    list[tuple[str, dict, float]]
        ``(document_text, metadata, cross_score)`` triples sorted by
        *cross_score* **descending** (most relevant first).
        Each metadata dict is augmented with a ``"cross_score"`` key.

    Fallback behaviour
    ------------------
    If no cross-encoder is registered (``set_cross_encoder`` was never
    called), the function returns documents in their original order with
    ``cross_score = 0.0`` and logs a warning.
    """
    if not documents:
        logger.debug("[RE_RANKER] No documents to re-rank — returning empty list")
        return []

    encoder = get_cross_encoder()

    if encoder is None:
        logger.warning(
            "[RE_RANKER] No cross-encoder model registered — "
            "returning documents in original order with cross_score=0.0"
        )
        results: List[Tuple[str, Dict[str, Any], float]] = []
        for doc, meta in zip(documents, metadatas):
            meta["cross_score_raw"] = 0.0
            meta["cross_score"] = 0.0
            results.append((doc, meta, 0.0))
        return results

    # ----- Build (query, passage) pairs for batch scoring ----- #
    pairs: List[Tuple[str, str]] = [(query, doc) for doc in documents]

    try:
        scores: List[float] = encoder.predict(pairs)
    except Exception as exc:
        logger.error(
            "[RE_RANKER] Cross-encoder scoring failed (%s) — "
            "returning documents in original order with cross_score=0.0",
            exc,
        )
        results = []
        for doc, meta in zip(documents, metadatas):
            meta["cross_score_raw"] = 0.0
            meta["cross_score"] = 0.0
            results.append((doc, meta, 0.0))
        return results

    # Defensive: ensure scores list matches documents length
    if len(scores) != len(documents):
        logger.error(
            "[RE_RANKER] Score count (%d) != document count (%d) — "
            "falling back to original order",
            len(scores),
            len(documents),
        )
        results = []
        for doc, meta in zip(documents, metadatas):
            meta["cross_score_raw"] = 0.0
            meta["cross_score"] = 0.0
            results.append((doc, meta, 0.0))
        return results

    # ----- Sigmoid normalization when scores are outside [0, 1] ----- #
    raw_scores = [float(s) for s in scores]

    if _needs_normalization(raw_scores):
        normalized_scores = [_sigmoid(s) for s in raw_scores]
        logger.info(
            "[RE_RANKER] Raw scores outside [0,1] detected — applying sigmoid normalization"
        )
    else:
        normalized_scores = list(raw_scores)
        logger.debug("[RE_RANKER] Scores already in [0,1] — no normalization needed")

    # ----- Inject both raw and normalized scores into metadata ----- #
    scored: List[Tuple[str, Dict[str, Any], float]] = []
    for doc, meta, raw, norm in zip(documents, metadatas, raw_scores, normalized_scores):
        meta["cross_score_raw"] = round(raw, 6)
        meta["cross_score"] = round(norm, 6)
        scored.append((doc, meta, round(norm, 6)))

    # ----- Sort by normalized cross_score descending ----- #
    scored.sort(key=lambda x: x[2], reverse=True)

    # ----- Structured logging ----- #
    sorted_raw = [s[1].get("cross_score_raw", 0.0) for s in scored]
    sorted_norm = [s[2] for s in scored]
    best_cross_score = sorted_norm[0] if sorted_norm else 0.0

    logger.info(
        "[RE_RANKER] cross_scores_raw: %s",
        [round(s, 4) for s in sorted_raw],
    )
    logger.info(
        "[RE_RANKER] cross_scores: %s",
        [round(s, 4) for s in sorted_norm],
    )
    logger.info("[RE_RANKER] best_cross_score: %.6f (normalized)", best_cross_score)

    return scored
