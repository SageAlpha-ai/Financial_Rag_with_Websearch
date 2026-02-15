"""
Immutable Retrieval Bundle — governance abstraction for retrieval state.

``RetrievalBundle`` wraps the three outputs of a retrieval step (documents,
metadatas, metrics) in a frozen, property-only container.  This prevents
accidental destructive mutation of retrieval evidence by downstream stages
(adequacy evaluation, routing, answer generation).

Design invariants:
    * ``documents`` and ``metadatas`` are exposed as **tuples** (immutable).
    * ``metrics`` is returned as a **shallow copy** on every access.
    * No public setter, ``__setattr__`` override, or mutation method exists.
    * All construction happens in ``__init__``; the object is sealed
      immediately after.

Usage::

    bundle = RetrievalBundle(docs, metas, metrics)
    docs   = list(bundle.documents)   # safe mutable copy
    metas  = list(bundle.metadatas)   # safe mutable copy
    m      = bundle.metrics           # safe dict copy
"""

from typing import Any, Dict, List, Tuple


class RetrievalBundle:
    """Immutable container for a single retrieval step's output.

    Parameters
    ----------
    documents : list[str]
        Retrieved document texts.
    metadatas : list[dict]
        Per-document metadata dicts (source, fiscal_year, page, etc.).
    metrics : dict[str, Any]
        Retrieval-quality metrics (best_distance, trust_score, etc.).
    """

    __slots__ = ("_documents", "_metadatas", "_metrics")

    def __init__(
        self,
        documents: List[str],
        metadatas: List[Dict],
        metrics: Dict[str, Any],
    ):
        object.__setattr__(self, "_documents", tuple(documents))
        object.__setattr__(self, "_metadatas", tuple(metadatas))
        object.__setattr__(self, "_metrics", dict(metrics))

    # --- read-only properties -------------------------------------------

    @property
    def documents(self) -> Tuple[str, ...]:
        """Retrieved document texts (immutable tuple)."""
        return self._documents

    @property
    def metadatas(self) -> Tuple[Dict, ...]:
        """Per-document metadata dicts (immutable tuple)."""
        return self._metadatas

    @property
    def metrics(self) -> Dict[str, Any]:
        """Retrieval-quality metrics (shallow copy on every access)."""
        return dict(self._metrics)

    # --- prevent mutation -----------------------------------------------

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(
            f"RetrievalBundle is immutable — cannot set '{name}'"
        )

    def __delattr__(self, name: str) -> None:
        raise AttributeError(
            f"RetrievalBundle is immutable — cannot delete '{name}'"
        )

    # --- convenience ----------------------------------------------------

    def __len__(self) -> int:
        """Number of retrieved documents."""
        return len(self._documents)

    def __bool__(self) -> bool:
        """True when at least one document was retrieved."""
        return len(self._documents) > 0

    def __repr__(self) -> str:
        return (
            f"RetrievalBundle(documents={len(self._documents)}, "
            f"metadatas={len(self._metadatas)}, "
            f"metrics_keys={sorted(self._metrics.keys())})"
        )
