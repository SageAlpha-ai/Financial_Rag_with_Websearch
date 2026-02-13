"""
LLM token and cost telemetry.

Pure observability: records token usage and estimated cost per request.
No business logic. Config-driven. Safe when usage or pricing is missing.
"""

import contextvars
import logging
from typing import Any, Dict, Optional

from config.settings import get_config
from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)

# Request-scoped stage for callback (which LLM call we are in).
_llm_cost_stage: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "llm_cost_stage", default=None
)


def set_llm_cost_stage(stage: str) -> None:
    _llm_cost_stage.set(stage)


def get_llm_cost_stage() -> Optional[str]:
    return _llm_cost_stage.get(None)


def _extract_usage_from_llm_result(response: Any) -> Optional[Dict[str, int]]:
    """Safely extract token usage from LangChain LLMResult. Returns None if missing."""
    try:
        # llm_output from OpenAI/Azure often has token_usage
        out = getattr(response, "llm_output", None) or {}
        if isinstance(out, dict):
            usage = out.get("token_usage") or out.get("usage")
            if isinstance(usage, dict):
                return usage
        # Some integrations put usage in the first generation's message
        gens = getattr(response, "generations", None) or []
        if gens and len(gens) > 0:
            gen = gens[0]
            if hasattr(gen, "message"):
                msg = gen.message
                meta = getattr(msg, "response_metadata", None) or {}
                usage = meta.get("usage") or meta.get("token_usage")
                if isinstance(usage, dict):
                    return usage
    except Exception:
        pass
    return None


def _record_usage_from_response(response: Any) -> None:
    """If response has usage, record to cost_tracker. Never raises."""
    try:
        usage = _extract_usage_from_llm_result(response)
        if usage is None:
            return
        stage = get_llm_cost_stage() or "unknown"
        model = "gpt-4o-mini"
        try:
            model = get_config().azure_openai.large_chat_deployment
        except Exception:
            pass
        inp = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        out = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        cost_tracker.record(stage=stage, model=model, input_tokens=inp, output_tokens=out)
    except Exception:
        logger.exception("cost_tracker_record_failed")


class CostTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self._data = {
            "stages": {},
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost_usd": 0.0,
        }

    def record(
        self,
        stage: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ):
        try:
            cfg = get_config().llm_cost
            if not cfg.get("enabled", False):
                return

            pricing = cfg.get("pricing", {}).get(model, {})
            input_price = pricing.get("input", 0.0)
            output_price = pricing.get("output", 0.0)

            cost = (input_tokens * input_price) + (
                output_tokens * output_price
            )

            self._data["stages"][stage] = {
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost,
            }

            self._data["total_input_tokens"] += input_tokens
            self._data["total_output_tokens"] += output_tokens
            self._data["total_cost_usd"] += cost

        except Exception:
            logger.exception("cost_tracker_record_failed")

    def summary(self) -> Dict[str, Any]:
        return dict(self._data)


cost_tracker = CostTracker()


class _CostTrackerCallback(BaseCallbackHandler):
    """LangChain callback to record LLM token usage from chain.invoke()."""

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        try:
            _record_usage_from_response(response)
        except Exception:
            logger.exception("cost_tracker_record_failed")


# Single shared instance for use with config={"callbacks": [cost_tracker_callback]}
cost_tracker_callback = _CostTrackerCallback()
