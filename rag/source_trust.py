"""
Source Trust Scoring — institutional-grade document reliability weights.

Assigns a deterministic trust score (0.0–1.0) to every document based
**solely** on structured metadata fields:

* ``document_type`` / ``source_type`` — for internal (RAG) documents
* ``domain`` / ``url``                — for web-sourced documents

No content inspection, no keyword heuristics, no LLM calls.

Trust weights are loaded from ``config.settings.SOURCE_TRUST_CONFIG`` and
can be overridden without code changes.

Public API
----------
``SourceTrustScorer.score(meta)``
    → ``float`` in [0.0, 1.0].

``SourceTrustScorer.score_internal(meta)``
    → trust weight for an internal / RAG document.

``SourceTrustScorer.score_web(meta)``
    → trust weight for a web-sourced document.
"""

import logging
from typing import Any, Dict
from urllib.parse import urlparse

from config.settings import SOURCE_TRUST_CONFIG

logger = logging.getLogger(__name__)

# Pre-resolve config sub-dicts once at import time.
_INTERNAL_TRUST: Dict[str, float] = SOURCE_TRUST_CONFIG.get("internal", {})
_WEB_TRUST: Dict[str, float] = SOURCE_TRUST_CONFIG.get("web", {})

_INTERNAL_DEFAULT: float = _INTERNAL_TRUST.get("default", 0.75)
_WEB_DEFAULT: float = _WEB_TRUST.get("unknown", 0.50)

# ---------------------------------------------------------------------------
# Well-known domain → trust-tier mapping (config-driven)
# ---------------------------------------------------------------------------
# The config stores coarse tiers ("official_domain", "regulator_domain",
# "financial_news").  The mapping below resolves concrete domains to those
# tiers.  Add new domains here — the *trust weight* stays in config.

_REGULATOR_DOMAINS = frozenset([
    "sec.gov", "sebi.gov.in", "rbi.org.in", "nseindia.com",
    "bseindia.com", "services.india.gov.in",
])

_FINANCIAL_NEWS_DOMAINS = frozenset([
    "reuters.com", "bloomberg.com", "moneycontrol.com",
    "economictimes.indiatimes.com", "livemint.com",
    "cnbc.com", "ft.com", "wsj.com",
])


def _extract_domain(meta: Dict[str, Any]) -> str:
    """Return a bare domain from metadata, or empty string."""
    domain = meta.get("domain", "")
    if domain:
        return domain.lower().strip()
    url = meta.get("url", "")
    if url:
        try:
            parsed = urlparse(url)
            return (parsed.netloc or "").lower().strip()
        except Exception:
            return ""
    return ""


def _domain_tier(domain: str) -> str:
    """Classify *domain* into a config tier key."""
    if not domain:
        return "unknown"
    for reg in _REGULATOR_DOMAINS:
        if domain == reg or domain.endswith("." + reg):
            return "regulator_domain"
    for news in _FINANCIAL_NEWS_DOMAINS:
        if domain == news or domain.endswith("." + news):
            return "financial_news"
    # If the domain looks like an official company site (not news, not
    # regulator) we treat it as an official domain.  This covers IR
    # pages hosted on corporate websites.
    return "official_domain"


class SourceTrustScorer:
    """Deterministic, config-driven source trust scorer."""

    # ------------------------------------------------------------------ #
    # Internal (RAG) documents                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def score_internal(meta: Dict[str, Any]) -> float:
        """Return trust weight for an internal / RAG document.

        Lookup order:
        1. ``meta["document_type"]``  → config ``internal[document_type]``
        2. ``meta["source_type"]``    → config ``internal[source_type]``
        3. Fallback                    → config ``internal["default"]``
        """
        doc_type = str(meta.get("document_type", "")).lower().strip()
        if doc_type and doc_type in _INTERNAL_TRUST:
            return float(_INTERNAL_TRUST[doc_type])

        src_type = str(meta.get("source_type", "")).lower().strip()
        if src_type and src_type in _INTERNAL_TRUST:
            return float(_INTERNAL_TRUST[src_type])

        return _INTERNAL_DEFAULT

    # ------------------------------------------------------------------ #
    # Web documents                                                       #
    # ------------------------------------------------------------------ #
    @staticmethod
    def score_web(meta: Dict[str, Any]) -> float:
        """Return trust weight for a web-sourced document.

        Resolves the document's domain to a config tier
        (``regulator_domain``, ``official_domain``, ``financial_news``,
        ``unknown``) and returns the corresponding weight.
        """
        domain = _extract_domain(meta)
        tier = _domain_tier(domain)
        return float(_WEB_TRUST.get(tier, _WEB_DEFAULT))

    # ------------------------------------------------------------------ #
    # Unified scorer                                                      #
    # ------------------------------------------------------------------ #
    @staticmethod
    def score(meta: Dict[str, Any]) -> float:
        """Return trust weight for any document.

        Dispatches to :meth:`score_web` when ``meta`` contains a web
        indicator (``url`` or ``domain`` key), otherwise falls through
        to :meth:`score_internal`.

        The resulting score is clamped to [0.0, 1.0].
        """
        # Web documents always carry a URL or domain field.
        is_web = bool(meta.get("url")) or bool(meta.get("domain"))

        if is_web:
            raw = SourceTrustScorer.score_web(meta)
        else:
            raw = SourceTrustScorer.score_internal(meta)

        trust = max(0.0, min(1.0, raw))

        logger.info(
            "[TRUST] doc_id=%s trust_score=%.4f source_type=%s",
            meta.get("source", meta.get("url", "unknown"))[:60],
            trust,
            "web" if is_web else "internal",
        )
        return trust
