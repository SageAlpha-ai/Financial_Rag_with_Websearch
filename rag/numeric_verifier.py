"""
Multi-Source Numeric Verification — evidence-layer validation.

Extracts numeric values from **semantically relevant** sentences in
retrieved documents, groups them by approximate equality, and reports
whether the sources agree on a value, provide only a single data
point, or are in conflict.

Semantic filtering is driven by the **cross-encoder** model registered
in ``rag.cross_encoder_reranker``.  Each sentence in a document is
scored against the user's question; only high-relevance sentences are
used for numeric extraction.  This eliminates page numbers, dates,
footnote markers, and unrelated figures without any keyword heuristics.

When no cross-encoder is available, the module falls back to
full-document extraction (every number in the document is considered).

This module sits between retrieval/re-ranking and answer generation.
It does **not** modify the answerability model, the planner, or the
retrieval pipeline.  It only produces a verification verdict that the
orchestrator can act on.

Public API
----------
``verify_numeric_consistency(question, documents, metadatas)``
    → ``Dict`` with ``status``, ``agreement_count``,
      ``distinct_values_count``, ``top_value``, ``trust_weight_sum``.
"""

import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sigmoid normalization (shared with cross_encoder_reranker)
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid: 1 / (1 + exp(-x))."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _needs_normalization(scores: List[float]) -> bool:
    """Return True when any score falls outside [0, 1]."""
    return any(s < 0.0 or s > 1.0 for s in scores)


# ---------------------------------------------------------------------------
# Semantic sentence-level relevance filtering (cross-encoder driven)
# ---------------------------------------------------------------------------

def _semantic_relevant_sentences(
    question: str,
    text: str,
    top_k: int = 5,
    threshold: float = 0.5,
) -> List[str]:
    """Return sentences from *text* that are semantically relevant to *question*.

    Uses the cross-encoder registered in ``rag.cross_encoder_reranker`` to
    score each sentence against the question.

    Parameters
    ----------
    question : str
        The user query.
    text : str
        Full document text to split into sentences.
    top_k : int
        Fallback count: if no sentence meets *threshold*, keep the
        *top_k* highest-scoring sentences instead.
    threshold : float
        Minimum normalised cross-encoder score for a sentence to be
        considered relevant.

    Returns
    -------
    list[str]
        Selected sentences (order preserved from original text).
        If no cross-encoder is available, returns **all** non-empty
        sentences (full-document fallback).
    """
    # --- Split into sentences ---
    raw_sentences = re.split(r"[.\n;]+", text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]

    if not sentences:
        return []

    # --- Obtain cross-encoder ---
    from rag.cross_encoder_reranker import get_cross_encoder

    model = get_cross_encoder()

    if model is None:
        # No cross-encoder registered — return all sentences (full-doc fallback)
        logger.debug(
            "[NUMERIC_VERIFIER] No cross-encoder available — "
            "returning all %d sentences (full-document fallback)",
            len(sentences),
        )
        return sentences

    # --- Score sentences against the question ---
    pairs: List[Tuple[str, str]] = [(question, sent) for sent in sentences]

    try:
        raw_scores: List[float] = [float(s) for s in model.predict(pairs)]
    except Exception as exc:
        logger.warning(
            "[NUMERIC_VERIFIER] Cross-encoder scoring failed (%s) — "
            "falling back to full-document extraction",
            exc,
        )
        return sentences

    if len(raw_scores) != len(sentences):
        logger.warning(
            "[NUMERIC_VERIFIER] Score count (%d) != sentence count (%d) — "
            "falling back to full-document extraction",
            len(raw_scores),
            len(sentences),
        )
        return sentences

    # --- Normalize scores to [0, 1] ---
    if _needs_normalization(raw_scores):
        scores = [_sigmoid(s) for s in raw_scores]
    else:
        scores = list(raw_scores)

    # --- Select relevant sentences ---
    scored: List[Tuple[str, float, int]] = [
        (sent, score, idx)
        for idx, (sent, score) in enumerate(zip(sentences, scores))
    ]

    # Sort by score descending for selection
    scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)

    # Sentences meeting the threshold
    above_threshold = [s for s in scored_sorted if s[1] >= threshold]

    if above_threshold:
        selected_set = set(s[2] for s in above_threshold)
    else:
        # No sentence meets threshold — keep top_k highest
        selected_set = set(s[2] for s in scored_sorted[:top_k])

    # Preserve original document order
    selected = [sentences[i] for i in sorted(selected_set)]

    # --- Structured logging ---
    top_5_scores = [round(s[1], 4) for s in scored_sorted[:5]]

    logger.info(
        "[NUMERIC_VERIFIER] semantic_sentences_total: %d",
        len(sentences),
    )
    logger.info(
        "[NUMERIC_VERIFIER] semantic_top_scores: %s",
        top_5_scores,
    )
    logger.info(
        "[NUMERIC_VERIFIER] semantic_selected_count: %d",
        len(selected),
    )

    return selected


# ---------------------------------------------------------------------------
# Numeric extraction
# ---------------------------------------------------------------------------

# Matches numbers with optional currency symbol, commas, decimals, and
# optional trailing %/M/B/K/Cr/L suffixes common in financial text.
#
# Examples matched:
#   $1,234.56   ₹12,345.67   1234   12.5%   3.2B   450Cr   12,500M
_NUMERIC_RE = re.compile(
    r"(?:[\$€£₹¥])\s*"                   # optional leading currency symbol
    r"(?P<number>[\d,]+(?:\.\d+)?)"       # digits with optional commas/decimal
    r"(?:\s*(?:million|billion|crore|lakh|mn|bn|cr|l|m|b|k|%))?"  # suffix
    r"|"
    r"(?P<plain>[\d,]+(?:\.\d+)?)"        # plain number (no currency)
    r"(?:\s*(?:million|billion|crore|lakh|mn|bn|cr|l|m|b|k|%))?",
    re.IGNORECASE,
)


def _parse_float(raw: str) -> Optional[float]:
    """Convert a comma-formatted numeric string to float, or None."""
    cleaned = raw.replace(",", "").strip()
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _contains_numeric_token(text: str) -> bool:
    """Return True if *text* contains at least one numeric pattern.

    Reuses the module-level ``_NUMERIC_RE`` — the same regex used by
    ``_extract_numbers`` — so the gate is structurally consistent with
    extraction.  No keyword heuristics are involved.
    """
    return bool(_NUMERIC_RE.search(text))


def _extract_numbers(text: str) -> List[float]:
    """Return all parseable numeric values found in *text*."""
    values: List[float] = []
    for match in _NUMERIC_RE.finditer(text):
        raw = match.group("number") or match.group("plain")
        if raw:
            val = _parse_float(raw)
            if val is not None and val != 0.0:
                values.append(val)
    return values


# ---------------------------------------------------------------------------
# Grouping by approximate equality
# ---------------------------------------------------------------------------

def _values_match(a: float, b: float, tolerance: float = 0.005) -> bool:
    """Return True when *a* and *b* are within *tolerance* relative error.

    Uses ``max(|a|, |b|)`` as the denominator to avoid division-by-zero
    and to be symmetric.  Default tolerance is 0.5 % — suitable for
    financial figures that may be rounded differently across sources.
    """
    if a == b:
        return True
    denom = max(abs(a), abs(b))
    if denom == 0.0:
        return True
    return abs(a - b) / denom <= tolerance


def _group_trust_sum(group: List[Dict[str, Any]]) -> float:
    """Return the total trust weight of a candidate group."""
    return sum(c.get("trust_score", 0.0) for c in group)


def _group_values(
    candidates: List[Dict[str, Any]],
    tolerance: float = 0.005,
) -> List[List[Dict[str, Any]]]:
    """Group numeric candidates by approximate equality.

    Each candidate is ``{"value": float, "source_index": int,
    "trust_score": float}``.

    Returns a list of groups (each group is a list of candidates whose
    values are all within *tolerance* of the group's first member).
    Groups are sorted by **total trust weight** (descending) so that
    the most institutionally reliable group appears first.
    """
    groups: List[List[Dict[str, Any]]] = []
    for cand in candidates:
        placed = False
        for group in groups:
            if _values_match(group[0]["value"], cand["value"], tolerance):
                group.append(cand)
                placed = True
                break
        if not placed:
            groups.append([cand])

    groups.sort(key=_group_trust_sum, reverse=True)
    return groups


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def verify_numeric_consistency(
    question: str,
    documents: List[str],
    metadatas: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Check whether retrieved documents agree on numeric values.

    Extraction is restricted to **semantically relevant** sentences as
    scored by the cross-encoder model.  Only sentences whose relevance
    to the question exceeds a threshold are used for number extraction.
    When no cross-encoder is available, falls back to full-document
    extraction.

    Parameters
    ----------
    question : str
        The user query — used by the cross-encoder to score sentence
        relevance.
    documents : list[str]
        Retrieved document texts.
    metadatas : list[dict]
        Per-document metadata dicts (parallel to *documents*).

    Returns
    -------
    dict
        ::

            {
                "status":                str,   # CONSISTENT | SINGLE_SOURCE | CONFLICT
                "agreement_count":       int,   # size of largest value group
                "distinct_values_count": int,   # number of distinct value groups
                "top_value":             float | None,  # most-agreed-upon value
                "trust_weight_sum":      float, # total trust of top group
                "groups":                list,  # per-group summaries (trust-sorted)
            }

    Status semantics
    ----------------
    ``CONSISTENT``
        Two or more sources contain the same numeric value (within 0.5 %
        tolerance).

    ``SINGLE_SOURCE``
        Only one source contains a numeric value (cannot cross-verify).

    ``CONFLICT``
        Multiple sources contain numeric values, but no two agree.
    """
    # Limit to top 5 documents
    top_docs = documents[:5]
    top_metas = metadatas[:5]

    # --- Extract numeric candidates from semantically relevant sentences ---
    candidates: List[Dict[str, Any]] = []
    for idx, doc in enumerate(top_docs):
        relevant_sentences = _semantic_relevant_sentences(question, doc)

        if not relevant_sentences:
            logger.debug(
                "[NUMERIC_VERIFIER] Doc %d: no relevant sentences — skipped",
                idx,
            )
            continue

        # --- Structural numeric presence gating ---
        # Prefer sentences that actually contain numeric tokens so that
        # purely descriptive (but semantically relevant) text does not
        # dilute extraction.  If *no* selected sentence carries a number,
        # fall back to the full selected set to avoid silent data loss.
        numeric_sentences = [
            s for s in relevant_sentences if _contains_numeric_token(s)
        ]

        _fallback = len(numeric_sentences) == 0
        extraction_sentences = numeric_sentences if not _fallback else relevant_sentences

        logger.info(
            "[NUMERIC_VERIFIER] Doc %d: numeric_sentence_count=%d fallback=%s",
            idx,
            len(numeric_sentences),
            str(_fallback),
        )

        filtered_text = " ".join(extraction_sentences)

        # Resolve trust score from metadata (attached by orchestrator).
        _doc_trust = float(
            top_metas[idx].get("trust_score", 0.75) if idx < len(top_metas) else 0.75
        )

        for val in _extract_numbers(filtered_text):
            candidates.append({
                "value": val,
                "source_index": idx,
                "trust_score": _doc_trust,
            })

    logger.debug(
        "[NUMERIC_VERIFIER] candidates_extracted=%d",
        len(candidates),
    )

    if not candidates:
        logger.debug("[NUMERIC_VERIFIER] No numeric values found in top 5 docs")
        return {
            "status": "CONSISTENT",
            "agreement_count": 0,
            "distinct_values_count": 0,
            "top_value": None,
            "trust_weight_sum": 0.0,
            "groups": [],
        }

    # --- Group by approximate equality (sorted by trust weight) ---
    groups = _group_values(candidates, tolerance=0.005)

    agreement_count = len(groups[0]) if groups else 0
    distinct_values_count = len(groups)
    top_value = round(groups[0][0]["value"], 4) if groups else None

    # --- Trust-weighted agreement ---
    top_trust_sum = round(_group_trust_sum(groups[0]), 4) if groups else 0.0

    trust_group_scores = [
        round(_group_trust_sum(g), 4) for g in groups
    ]

    logger.info(
        "[NUMERIC_VERIFIER] trust_group_scores=%s",
        trust_group_scores,
    )

    # --- Determine status via trust thresholds ---
    if top_trust_sum >= 1.5:
        status = "CONSISTENT"
    elif top_trust_sum < 1.0:
        status = "SINGLE_SOURCE"
    else:
        # Check if competing groups are within 0.2 margin of top group
        if len(groups) >= 2:
            second_trust = _group_trust_sum(groups[1])
            if top_trust_sum - second_trust <= 0.2:
                status = "CONFLICT"
            else:
                status = "CONSISTENT"
        else:
            status = "SINGLE_SOURCE"

    # --- Build per-group summary (sorted by trust_weight_sum desc) ---
    group_summaries: List[Dict[str, Any]] = []
    for g in groups:
        g_sources = set(c["source_index"] for c in g)
        group_summaries.append({
            "value": round(g[0]["value"], 4),
            "trust_weight_sum": round(_group_trust_sum(g), 4),
            "source_count": len(g_sources),
        })
    # groups are already sorted by trust weight; summaries inherit order.

    if status == "CONFLICT":
        logger.warning(
            "[NUMERIC_VERIFIER] conflict_groups=%s",
            group_summaries,
        )

    result = {
        "status": status,
        "agreement_count": agreement_count,
        "distinct_values_count": distinct_values_count,
        "top_value": top_value,
        "trust_weight_sum": top_trust_sum,
        "groups": group_summaries,
    }

    logger.info(
        "[NUMERIC_VERIFIER] status=%s  agreement_count=%d  "
        "distinct_values=%d  top_value=%s  trust_weight_sum=%.4f",
        status,
        agreement_count,
        distinct_values_count,
        top_value,
        top_trust_sum,
    )

    return result
