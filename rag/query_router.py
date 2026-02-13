"""
LLM-Based Query Router

Uses Azure OpenAI GPT-4o-mini to semantically classify a user query into
one of three routing categories:

    - "rag"    — answerable from internal indexed documents
    - "web"    — requires fresh or external public data
    - "hybrid" — needs both internal documents AND web sources

Also determines query scope:

    - "single_company" — targets one company
    - "multi_company"  — compares multiple companies
    - "sector"         — sector-wide analysis
    - "macro"          — macroeconomic question

Returns a structured JSON result::

    {
        "route":      "rag" | "web" | "hybrid",
        "scope":      "single_company" | "multi_company" | "sector" | "macro",
        "confidence": 0.0–1.0,
        "reason":     "<short explanation>"
    }

Design principles:
    - Deterministic: temperature=0, JSON output mode
    - Safe: defaults to "rag" on any failure (LLM error, malformed output, etc.)
    - Follows existing Azure OpenAI client patterns (see rag/planner.py)
    - Lightweight: reuses the planner mini-model deployment

Environment variables:
    AZURE_OPENAI_ROUTER_CHAT_DEPLOYMENT_NAME  (optional)
        Falls back to AZURE_OPENAI_PLANNER_CHAT_DEPLOYMENT_NAME if not set.
"""

import json
import logging
import os
from typing import Any, Dict

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.settings import get_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_ROUTES: set[str] = {"rag", "web", "hybrid"}
VALID_SCOPES: set[str] = {"single_company", "multi_company", "sector", "macro"}

DEFAULT_ROUTE: Dict[str, Any] = {
    "route": "rag",
    "scope": "single_company",
    "confidence": 0.0,
    "reason": "Routing failed; defaulting to RAG for safety.",
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT: str = (
    "You are a financial query routing system.\n"
    "\n"
    "Your job is to determine how the query should be handled.\n"
    "\n"
    "Available routes:\n"
    '- "rag" → Use internal company financial documents (vector database)\n'
    '- "web" → Requires live or external information\n'
    '- "hybrid" → Requires both internal documents and web data\n'
    "\n"
    "Classify the query based on:\n"
    "\n"
    "1. Does it reference a specific company?\n"
    "2. Does it compare multiple companies?\n"
    "3. Does it require current/live market data?\n"
    "4. Is it macroeconomic or sector-wide?\n"
    "5. Is the answer likely stored in internal filings?\n"
    "\n"
    "Return ONLY valid JSON:\n"
    "\n"
    "{\n"
    '  "route": "rag" | "web" | "hybrid",\n'
    '  "scope": "single_company" | "multi_company" | "sector" | "macro",\n'
    '  "confidence": <float 0.0 to 1.0>,\n'
    '  "reason": "one sentence explanation"\n'
    "}\n"
)

# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------


def _get_router_deployment() -> str:
    """Resolve the Azure OpenAI deployment name for routing.

    Checks ``AZURE_OPENAI_ROUTER_CHAT_DEPLOYMENT_NAME`` first, then falls
    back to ``AZURE_OPENAI_PLANNER_CHAT_DEPLOYMENT_NAME`` (both use
    GPT-4o-mini in the default configuration).

    Raises:
        ValueError: If neither environment variable is set.
    """
    deployment = os.getenv("AZURE_OPENAI_ROUTER_CHAT_DEPLOYMENT_NAME")
    if deployment:
        return deployment

    deployment = os.getenv("AZURE_OPENAI_PLANNER_CHAT_DEPLOYMENT_NAME")
    if deployment:
        return deployment

    raise ValueError(
        "Missing deployment name. Set AZURE_OPENAI_ROUTER_CHAT_DEPLOYMENT_NAME "
        "or AZURE_OPENAI_PLANNER_CHAT_DEPLOYMENT_NAME."
    )


def _build_router_llm() -> AzureChatOpenAI:
    """Build the Azure OpenAI chat client for query routing.

    Follows the same pattern as ``rag.planner._build_planner_llm()``:
    reuses shared Azure OpenAI credentials from ``config.settings`` and
    enables JSON output mode with temperature 0 for deterministic results.

    Returns:
        A configured ``AzureChatOpenAI`` instance.

    Raises:
        ValueError: If the deployment name environment variable is missing.
    """
    config = get_config()
    deployment = _get_router_deployment()

    llm = AzureChatOpenAI(
        azure_endpoint=config.azure_openai.endpoint,
        azure_deployment=deployment,
        api_key=config.azure_openai.api_key,
        api_version=config.azure_openai.api_version,
        temperature=0.0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

    logger.info(
        "[ROUTER-LLM] Client initialised (deployment=%s, temperature=0, json_mode=on)",
        deployment,
    )
    return llm


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_route_response(parsed: Any) -> Dict[str, Any]:
    """Validate and normalise the LLM's routing response.

    Ensures the response contains the four required fields with correct
    types and value ranges.  Missing or out-of-range values are coerced
    to safe defaults rather than raising — this keeps the router
    maximally resilient.

    Args:
        parsed: The object produced by ``json.loads`` on the LLM output.

    Returns:
        A clean ``{"route", "scope", "confidence", "reason"}`` dict.
    """
    if not isinstance(parsed, dict):
        logger.warning("[ROUTER-LLM] Response is not a dict: %s", type(parsed).__name__)
        return dict(DEFAULT_ROUTE)

    # --- route ---
    route = str(parsed.get("route", "rag")).lower().strip()
    if route not in VALID_ROUTES:
        logger.warning("[ROUTER-LLM] Invalid route '%s', defaulting to 'rag'", route)
        route = "rag"

    # --- scope ---
    scope = str(parsed.get("scope", "single_company")).lower().strip()
    if scope not in VALID_SCOPES:
        logger.warning("[ROUTER-LLM] Invalid scope '%s', defaulting to 'single_company'", scope)
        scope = "single_company"

    # --- confidence ---
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    # --- reason ---
    reason = str(parsed.get("reason", "No reason provided.")).strip()
    if not reason:
        reason = "No reason provided."

    return {
        "route": route,
        "scope": scope,
        "confidence": round(confidence, 4),
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_query_route(query: str) -> Dict[str, Any]:
    """Classify a user query into a routing category using GPT-4o-mini.

    This is the main entry point for LLM-based query routing.  It sends
    the query to Azure OpenAI with a tightly constrained system prompt
    and returns a validated routing decision.

    On ANY failure (network, auth, malformed output, unexpected exception),
    the function returns the safe default ``{"route": "rag", ...}`` so
    that the downstream pipeline can continue.

    Args:
        query: The raw user query string.

    Returns:
        A dict with exactly four keys::

            {
                "route":      "rag" | "web" | "hybrid",
                "scope":      "single_company" | "multi_company" | "sector" | "macro",
                "confidence": float between 0.0 and 1.0,
                "reason":     str
            }

    Example::

        >>> classify_query_route("What was Dixon's revenue in FY2024?")
        {"route": "rag", "scope": "single_company", "confidence": 0.95,
         "reason": "Fiscal year revenue is available in internal filings."}

        >>> classify_query_route("What is Tesla's stock price right now?")
        {"route": "web", "scope": "single_company", "confidence": 0.92,
         "reason": "Current stock price requires live market data."}

        >>> classify_query_route("Compare TCS and Infosys margins")
        {"route": "hybrid", "scope": "multi_company", "confidence": 0.88,
         "reason": "Multi-company comparison needs filings and current data."}
    """
    if not query or not query.strip():
        logger.warning("[ROUTER-LLM] Empty query received, defaulting to rag")
        return dict(DEFAULT_ROUTE)

    deployment = "unknown"
    try:
        deployment = _get_router_deployment()
    except ValueError:
        pass

    logger.info("[ROUTER-LLM] Routing started (model=%s, query_len=%d)", deployment, len(query))

    # -- Build LCEL chain --
    try:
        llm = _build_router_llm()
    except (ValueError, Exception) as exc:
        logger.error("[ROUTER-LLM] Failed to initialise LLM: %s", exc)
        return dict(DEFAULT_ROUTE)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            ("human", "{query}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()

    # -- Invoke --
    try:
        raw_output: str = chain.invoke({"query": query})
    except Exception as exc:
        logger.error("[ROUTER-LLM] LLM call failed: %s", exc)
        return dict(DEFAULT_ROUTE)

    raw_output = raw_output.strip()
    logger.debug("[ROUTER-LLM] Raw output: %s", raw_output)

    # -- Parse JSON --
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        logger.error("[ROUTER-LLM] Invalid JSON from LLM: %s", exc)
        return dict(DEFAULT_ROUTE)

    # -- Validate and normalise --
    result = _validate_route_response(parsed)

    logger.info(
        "[ROUTER-LLM] route=%s scope=%s confidence=%.4f reason=%s",
        result["route"],
        result["scope"],
        result["confidence"],
        result["reason"],
    )

    return result
