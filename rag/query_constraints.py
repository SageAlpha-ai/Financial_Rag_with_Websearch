"""
Query Constraint Extraction — single source of truth.

Centralises all lightweight, deterministic query-analysis helpers that
extract fiscal years, detect numeric intent, identify temporal queries,
and recognise system-introspection prompts.

Every function in this module is:
  - Pure (no side effects, no LLM calls, no I/O)
  - Deterministic (same input → same output)
  - Fast (regex / keyword matching only)

All other modules must import from here — local copies are prohibited.
"""

import re
from typing import Optional

# Re-export the LLM-backed company extractor so that callers needing
# both constraint helpers and company extraction can use a single import
# source.  The implementation lives in rag.company_extractor and is NOT
# duplicated here.
from rag.company_extractor import extract_company_name  # noqa: F401


# ---------------------------------------------------------------------------
# Fiscal-year extraction
# ---------------------------------------------------------------------------

def extract_fiscal_year(query: str) -> Optional[str]:
    """Extract fiscal year from *query*.

    Returns ``"FYxxxx"`` (e.g. ``"FY2023"``) or ``None`` if no year is found.

    Pattern priority (first match wins):
        1. Explicit ``FY 2023`` / ``FY2023``
        2. ``fiscal year 2023``
        3. Any bare 20xx year
        4. Any bare 19xx year
    """
    patterns = [
        r'FY\s*(\d{4})',
        r'fiscal\s+year\s+(\d{4})',
        r'\b(20\d{2})\b',
        r'\b(19\d{2})\b',
    ]
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return f"FY{match.group(1)}"
    return None


def normalize_fiscal_year(year: str) -> str:
    """Strip ``FY`` prefix and return a normalised 4-digit year string.

    Examples::

        normalize_fiscal_year("FY2023")  # → "2023"
        normalize_fiscal_year("2023")    # → "2023"
        normalize_fiscal_year("FY 2023") # → "2023"
    """
    stripped = re.sub(r'^FY\s*', '', year, flags=re.IGNORECASE).strip()
    return stripped


# ---------------------------------------------------------------------------
# Numeric / financial intent detection
# ---------------------------------------------------------------------------

def detect_numeric_intent(query: str) -> bool:
    """Return ``True`` when the query carries numeric or exact-value intent.

    Counts matches against a set of lightweight regex patterns.  A score
    of **>= 2** indicates the query is asking for concrete numbers.

    The pattern list and threshold are intentionally kept in sync with the
    original ``_detect_numeric_intent`` in ``langchain_orchestrator.py``.
    """
    query_lower = query.lower()
    numeric_patterns = [
        r'\d+', r'%', r'\$', r'\btotal\b', r'\bexact\b', r'\brate\b',
        r'\bvalue\b', r'\brevenue\b', r'\bamount\b', r'\bnumber\b', r'\bcount\b',
    ]
    numeric_score = sum(
        1 for pattern in numeric_patterns
        if re.search(pattern, query_lower, re.IGNORECASE)
    )
    return numeric_score >= 2


# ---------------------------------------------------------------------------
# Temporal intent detection
# ---------------------------------------------------------------------------

def detect_temporal_intent(query: str) -> bool:
    """Return ``True`` when the query requests present-moment / real-time data.

    Simple keyword matching — no LLM call.
    """
    _TEMPORAL_KEYWORDS = [
        "today", "current", "now", "latest", "as of",
        "present", "current time", "current date",
    ]
    query_lower = query.lower()
    return any(kw in query_lower for kw in _TEMPORAL_KEYWORDS)


# ---------------------------------------------------------------------------
# System introspection detection
# ---------------------------------------------------------------------------

def detect_system_introspection(query: str) -> bool:
    """Return ``True`` when the query asks about the system's own capabilities.

    Matches phrases about web/internet access, browsing capability,
    real-time data access, or training-data scope.  Matched queries are
    answered with a deterministic system-authored response so that no
    planner, RAG, web search, or LLM call is needed.
    """
    _INTROSPECTION_PHRASES = [
        "do you have internet",
        "do you have web",
        "can you browse",
        "can you search the web",
        "can you search the internet",
        "can you access the internet",
        "can you access the web",
        "can you access websites",
        "do you have access to the internet",
        "do you have access to the web",
        "do you have web search",
        "do you have real-time",
        "do you have realtime",
        "do you have real time",
        "do you have live data",
        "can you access real-time",
        "can you access realtime",
        "can you access live data",
        "what data do you have access to",
        "what sources do you use",
        "what is your training data",
        "what are your capabilities",
        "what can you do",
        "are you connected to the internet",
        "are you connected to the web",
        "do you browse the web",
        "can you look up websites",
        "can you fetch web pages",
        "can you go online",
        "are you online",
        "training data",
        "training cutoff",
        "knowledge cutoff",
    ]
    query_lower = query.lower()
    return any(phrase in query_lower for phrase in _INTROSPECTION_PHRASES)
