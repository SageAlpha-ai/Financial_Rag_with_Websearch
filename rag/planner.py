"""
Query Planner Module

Uses a lightweight LLM (gpt-4o-mini via Azure OpenAI) to produce a structured
execution plan for a user query.

The planner decides WHICH tools to invoke and in WHAT order.
It does NOT answer the user's question.

RESPONSIBILITY BOUNDARY
-----------------------
Planner output expresses INTENT and ORDER only.  It does NOT:
    - Guarantee that an answer exists in any data source.
    - Encode confidence or correctness of the eventual answer.
    - Encode fallback semantics beyond the declared tool order.

Answerability and evidence sufficiency are determined entirely downstream
by retrieval validation (e.g. entity/year/metric matching after documents
are fetched).  The planner is intentionally decoupled from those concerns.

Contract v1.1
-------------
{
    "analysis":       "<one-sentence reasoning about the query's needs>",
    "execution_plan": [
        { "tool": "RAG | WEB_SEARCH | LLM", "goal": "retrieve | answer" }
    ],
    "alternatives": [          // optional, 0-3 fallback plans
        [ { "tool": "...", "goal": "..." }, ... ],
        ...
    ],
    "reasoning":      "<short reasoning>"   // optional
}

Invariants (apply to execution_plan AND every alternative):
    - Plans are ordered (index 0 executes first).
    - Plans contain 1–7 steps.
    - The last step is always {"tool": "LLM", "goal": "answer"}.
    - RAG and WEB_SEARCH may only have goal "retrieve".
    - LLM may only have goal "answer".
    - No additional keys are permitted in step objects.

Backward compatibility:
    - ``alternatives`` and ``reasoning`` are optional.
    - Consumers that only read ``analysis`` and ``execution_plan`` are
      unaffected.
"""

import json
import logging
import os
from typing import Any, Dict, List

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.settings import get_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PlannerError(Exception):
    """Base exception for all planner failures."""


class PlannerValidationError(PlannerError):
    """Raised when LLM output violates the v1 contract.

    Attributes:
        raw_output: The verbatim string returned by the LLM before parsing.
    """

    def __init__(self, message: str, raw_output: str = "") -> None:
        self.raw_output = raw_output
        super().__init__(message)


# ---------------------------------------------------------------------------
# Contract constants
# ---------------------------------------------------------------------------

VALID_TOOLS: set[str] = {"RAG", "WEB_SEARCH", "LLM"}
VALID_GOALS: set[str] = {"retrieve", "answer"}

#: Every (tool, goal) pair that is permitted by the contract.
VALID_TOOL_GOAL_PAIRS: set[tuple[str, str]] = {
    ("RAG", "retrieve"),
    ("WEB_SEARCH", "retrieve"),
    ("LLM", "answer"),
}

_MAX_STEPS = 7
_MAX_ALTERNATIVES = 3


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT: str = (
    "You are a query planner. Your ONLY job is to decide which tools are "
    "needed to answer a user query and in what order. "
    "You MUST NOT answer the query itself.\n"
    "\n"
    "Available tools:\n"
    "  RAG        – retrieve from internal indexed documents\n"
    "  WEB_SEARCH – retrieve from external public sources\n"
    "  LLM        – generate, reason, or format a final answer\n"
    "\n"
    "Respond with a single JSON object and absolutely nothing else.\n"
    "No markdown fences, no commentary, no extra whitespace — only raw JSON.\n"
    "\n"
    "JSON schema:\n"
    "{{\n"
    '  "analysis": "<one sentence: what information does this query need?>",\n'
    '  "requires_rag": true | false,\n'
    '  "requires_web": true | false,\n'
    '  "complexity_level": "low | medium | high",\n'
    '  "execution_plan": [\n'
    '    {{ "tool": "RAG | WEB_SEARCH | LLM", "goal": "retrieve | answer" }}\n'
    "  ],\n"
    '  "alternatives": [               // optional, 0-3 fallback plans\n'
    "    [\n"
    '      {{ "tool": "...", "goal": "..." }}\n'
    "    ]\n"
    "  ],\n"
    '  "reasoning": "<short reasoning>"  // optional\n'
    "}}\n"
    "\n"
    "Rules you MUST follow:\n"
    '1. "execution_plan" is an ordered array of steps.\n'
    "2. Each plan (primary or alternative) MUST contain between 1 and 7 steps.\n"
    '3. The LAST step of every plan MUST be {{ "tool": "LLM", "goal": "answer" }}.\n'
    '4. RAG and WEB_SEARCH steps MUST use goal "retrieve".\n'
    '5. LLM steps MUST use goal "answer".\n'
    "6. Do NOT include duplicate consecutive steps within a plan.\n"
    '7. "alternatives" is an optional array of at most 3 fallback plans.\n'
    "   Each alternative is itself an ordered array of steps following the\n"
    "   same rules as execution_plan.\n"
    '8. "reasoning" is optional — a short sentence explaining your choice.\n'
    "9. Do NOT add any keys to step objects beyond \"tool\" and \"goal\".\n"
    "\n"
    "Strategy guidance:\n"
    "- If the query asks for recent, latest, or comparative data across "
    "companies, prefer WEB_SEARCH before RAG in the primary plan.\n"
    "- When the query is clearly about fresh / public information unlikely "
    "to exist in internal documents, lead with WEB_SEARCH.\n"
    "- When you are uncertain whether internal documents will suffice, "
    "provide at least one alternative that includes WEB_SEARCH as a "
    "fallback retrieval strategy.\n"
    "- If the query requests exact numeric, financial, or fiscal-year-specific "
    "values (e.g. revenue, profit, FY data), prefer retrieval-based plans "
    "(RAG or WEB_SEARCH) over LLM-only plans whenever possible.\n"
    "- If the query requires real-time or present-moment information — "
    "keywords like \"today\", \"current\", \"now\", \"latest\", "
    "\"current date\", \"current time\" — the execution_plan MUST be:\n"
    '  [{{ "tool": "WEB_SEARCH", "goal": "retrieve" }}, '
    '{{ "tool": "LLM", "goal": "answer" }}]\n'
    "  NEVER use LLM-only for real-time queries. Real-time information "
    "requires external retrieval.\n"
    "- If the query requires an \"equity research report\" or any multi-section formal financial report, the execution_plan MUST include BOTH RAG and WEB_SEARCH to ensure internal data and current market context are both present.\n"
    "- Express compound flows as ordered steps. "
    'Never use composite tokens like "RAG+LLM".\n'
)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_step_sequence(
    steps: Any,
    label: str,
    raw_output: str,
) -> None:
    """Validate a single ordered step sequence against the contract.

    This helper is shared by the primary ``execution_plan`` and every
    entry inside ``alternatives``.

    Args:
        steps: The step array to validate.
        label: Human-readable label for error messages (e.g.
            ``"execution_plan"`` or ``"alternatives[1]"``).
        raw_output: Verbatim LLM string attached to errors.

    Raises:
        PlannerValidationError: On the first rule violation encountered.
    """
    if not isinstance(steps, list):
        raise PlannerValidationError(
            f'"{label}" must be an array',
            raw_output,
        )

    if len(steps) < 1:
        raise PlannerValidationError(
            f'"{label}" must contain at least one step',
            raw_output,
        )

    if len(steps) > _MAX_STEPS:
        raise PlannerValidationError(
            f'"{label}" must contain at most {_MAX_STEPS} steps, '
            f"got {len(steps)}",
            raw_output,
        )

    step_keys_allowed = {"tool", "goal"}
    prev_step: Dict[str, str] | None = None

    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            raise PlannerValidationError(
                f"{label} step {idx} must be a JSON object, "
                f"got {type(step).__name__}",
                raw_output,
            )

        step_extra = set(step.keys()) - step_keys_allowed
        if step_extra:
            raise PlannerValidationError(
                f"{label} step {idx} has unexpected keys: {step_extra}",
                raw_output,
            )

        step_missing = step_keys_allowed - set(step.keys())
        if step_missing:
            raise PlannerValidationError(
                f"{label} step {idx} is missing required keys: {step_missing}",
                raw_output,
            )

        tool: str = step["tool"]
        goal: str = step["goal"]

        if tool not in VALID_TOOLS:
            raise PlannerValidationError(
                f'{label} step {idx}: invalid tool "{tool}". '
                f"Allowed: {sorted(VALID_TOOLS)}",
                raw_output,
            )

        if goal not in VALID_GOALS:
            raise PlannerValidationError(
                f'{label} step {idx}: invalid goal "{goal}". '
                f"Allowed: {sorted(VALID_GOALS)}",
                raw_output,
            )

        if (tool, goal) not in VALID_TOOL_GOAL_PAIRS:
            raise PlannerValidationError(
                f'{label} step {idx}: tool "{tool}" cannot have goal "{goal}"',
                raw_output,
            )

        if prev_step is not None and step == prev_step:
            raise PlannerValidationError(
                f"{label} step {idx} is a duplicate of step {idx - 1}",
                raw_output,
            )
        prev_step = step

    # -- last-step invariant --
    last = steps[-1]
    if last.get("tool") != "LLM" or last.get("goal") != "answer":
        raise PlannerValidationError(
            f'{label}: last step must be {{"tool": "LLM", "goal": "answer"}}, '
            f"got {last}",
            raw_output,
        )


def _validate_plan(plan: Any, raw_output: str) -> Dict[str, Any]:
    """Validate a parsed planner output against the v1.1 contract.

    Args:
        plan: The object produced by ``json.loads`` on the LLM output.
        raw_output: The verbatim LLM string (attached to errors for debugging).

    Returns:
        The same ``plan`` dict, unchanged, if every rule passes.

    Raises:
        PlannerValidationError: On the first rule violation encountered.
    """
    # -- top-level type --
    if not isinstance(plan, dict):
        raise PlannerValidationError(
            f"Plan must be a JSON object, got {type(plan).__name__}",
            raw_output,
        )

    # -- top-level keys --
    required_keys = {"analysis", "execution_plan", "requires_rag", "requires_web", "complexity_level"}
    optional_keys = {"alternatives", "reasoning"}
    allowed_keys = required_keys | optional_keys

    extra = set(plan.keys()) - allowed_keys
    if extra:
        raise PlannerValidationError(
            f"Unexpected top-level keys: {extra}",
            raw_output,
        )

    missing = required_keys - set(plan.keys())
    if missing:
        raise PlannerValidationError(
            f"Missing required top-level keys: {missing}",
            raw_output,
        )

    # -- analysis --
    if not isinstance(plan["analysis"], str) or not plan["analysis"].strip():
        raise PlannerValidationError(
            '"analysis" must be a non-empty string',
            raw_output,
        )

    # -- reasoning (optional) --
    if "reasoning" in plan:
        if not isinstance(plan["reasoning"], str):
            raise PlannerValidationError(
                '"reasoning" must be a string if present',
                raw_output,
            )

    # -- primary execution_plan --
    _validate_step_sequence(plan["execution_plan"], "execution_plan", raw_output)

    # -- alternatives (optional) --
    if "alternatives" in plan:
        alts = plan["alternatives"]
        if not isinstance(alts, list):
            raise PlannerValidationError(
                '"alternatives" must be an array of plans',
                raw_output,
            )
        if len(alts) > _MAX_ALTERNATIVES:
            raise PlannerValidationError(
                f'"alternatives" must contain at most {_MAX_ALTERNATIVES} '
                f"plans, got {len(alts)}",
                raw_output,
            )
        for alt_idx, alt_plan in enumerate(alts):
            _validate_step_sequence(
                alt_plan, f"alternatives[{alt_idx}]", raw_output
            )

    return plan


# ---------------------------------------------------------------------------
# LLM client builder
# ---------------------------------------------------------------------------


def _build_planner_llm() -> AzureChatOpenAI:
    """Build the Azure OpenAI chat client for the planner.

    Reuses the shared Azure OpenAI credentials (endpoint, API key,
    API version) from ``config.settings`` and reads the deployment name
    from the ``AZURE_OPENAI_PLANNER_CHAT_DEPLOYMENT_NAME`` environment
    variable.

    Returns:
        A configured ``AzureChatOpenAI`` instance with temperature 0 and
        JSON output mode enabled.

    Raises:
        ValueError: If ``AZURE_OPENAI_PLANNER_CHAT_DEPLOYMENT_NAME`` is
            not set.
    """
    config = get_config()

    planner_deployment = os.getenv("AZURE_OPENAI_PLANNER_CHAT_DEPLOYMENT_NAME")
    if not planner_deployment:
        raise ValueError(
            "Missing required environment variable: "
            "AZURE_OPENAI_PLANNER_CHAT_DEPLOYMENT_NAME"
        )

    llm = AzureChatOpenAI(
        azure_endpoint=config.azure_openai.endpoint,
        azure_deployment=planner_deployment,
        api_key=config.azure_openai.api_key,
        api_version=config.azure_openai.api_version,
        temperature=0.0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

    logger.info(
        "[PLANNER] LLM client initialised "
        "(deployment=%s, temperature=0, json_mode=on)",
        planner_deployment,
    )
    return llm


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plan_query(query: str) -> Dict[str, Any]:
    """Produce an execution plan for a user query.

    Sends the query to a lightweight LLM that returns a strict-JSON plan
    describing which tools to invoke and in what order.  The LLM is
    instructed never to answer the query itself.

    The returned plan expresses *intent and tool order only*.  It encodes
    neither confidence nor correctness — execution success and
    answerability are determined downstream by retrieval validation.

    Args:
        query: The raw user query string.

    Returns:
        A validated plan dict conforming to the v1.1 contract::

            {
                "analysis": "Needs latest revenue for comparison.",
                "execution_plan": [
                    {"tool": "WEB_SEARCH", "goal": "retrieve"},
                    {"tool": "LLM",        "goal": "answer"}
                ],
                "alternatives": [
                    [
                        {"tool": "RAG",        "goal": "retrieve"},
                        {"tool": "WEB_SEARCH", "goal": "retrieve"},
                        {"tool": "LLM",        "goal": "answer"}
                    ]
                ],
                "reasoning": "Web-first because latest revenue is needed."
            }

        ``alternatives`` and ``reasoning`` are optional — consumers that
        only read ``analysis`` and ``execution_plan`` are unaffected.

    Raises:
        PlannerValidationError: If the LLM output does not conform to the
            v1.1 contract (malformed JSON, missing fields, rule violations).
        PlannerError: If the LLM call itself fails (network, auth, deployment).
    """
    planner_deployment = os.getenv(
        "AZURE_OPENAI_PLANNER_CHAT_DEPLOYMENT_NAME", "unknown"
    )
    logger.info("[PLANNER] Planning started (model=%s)", planner_deployment)

    # -- build LCEL chain --
    try:
        llm = _build_planner_llm()
    except ValueError as exc:
        raise PlannerError(f"Failed to initialise planner LLM: {exc}") from exc

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            ("human", "{query}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()

    # -- invoke --
    try:
        from rag.telemetry import cost_tracker_callback, set_llm_cost_stage
        set_llm_cost_stage("planner")
        raw_output: str = chain.invoke(
            {"query": query},
            config={"callbacks": [cost_tracker_callback]},
        )
    except Exception as exc:
        raise PlannerError(f"Planner LLM call failed: {exc}") from exc

    raw_output = raw_output.strip()

    # -- parse JSON --
    try:
        plan = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise PlannerValidationError(
            f"Planner output is not valid JSON: {exc}",
            raw_output,
        ) from exc

    # -- validate against contract --
    # NOTE: Validation enforces structural correctness of the plan only.
    # The plan does not encode confidence, correctness, or answerability.
    # Execution success and answerability are determined downstream by
    # retrieval validation (entity/year/metric matching).
    validated_plan = _validate_plan(plan, raw_output)

    # -- structured log --
    alternatives = validated_plan.get("alternatives", [])
    alt_count = len(alternatives)

    logger.info("[PLANNER] model=%s", planner_deployment)
    logger.info("[PLANNER] analysis=%s", validated_plan["analysis"])
    logger.info(
        "[PLANNER] execution_plan=%s",
        json.dumps(validated_plan["execution_plan"], separators=(",", ":")),
    )
    logger.info("[PLANNER] alternatives_count=%d", alt_count)
    if alt_count > 0:
        for alt_idx, alt_plan in enumerate(alternatives):
            logger.info(
                "[PLANNER] alternative[%d]=%s",
                alt_idx,
                json.dumps(alt_plan, separators=(",", ":")),
            )
    if validated_plan.get("reasoning"):
        logger.info("[PLANNER] reasoning=%s", validated_plan["reasoning"])

    return validated_plan
