"""
Budget Guard Module for B17 Population Simulation.

Enforces spend tracking, token counting, and a hard dollar ceiling so that an
unattended overnight elicitation run cannot quietly empty an API account.

Prices come from ``shared.llm_client.MODEL_REGISTRY`` (single source of truth),
imported lazily to avoid an import cycle.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _pricing(model: str) -> Dict[str, float]:
    """Per-1M-token input/output price for ``model``, from the model registry."""
    from .llm_client import model_info  # lazy: llm_client imports this module

    try:
        info = model_info(model)
    except KeyError:
        # Unknown model: assume a small-frontier price so we over- rather than
        # under-estimate spend. Never silently assume free.
        logger.warning("No price registered for %s; assuming $3/$15 per 1M.", model)
        return {"input": 3.00, "output": 15.00}
    return {"input": float(info["in"]), "output": float(info["out"])}


class BudgetExceededError(Exception):
    """Raised when an API call would exceed the configured dollar limit."""

    pass


@dataclass
class BudgetGuard:
    max_usd: float = 1200.00
    state_file: Optional[Path] = None
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cached_calls: int = 0
    total_live_calls: int = 0
    total_usd_spent: float = 0.0
    model_breakdown: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def __post_init__(self):
        if self.state_file and self.state_file.exists():
            self._load_state()

    def _load_state(self):
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.total_prompt_tokens = data.get("total_prompt_tokens", 0)
                self.total_completion_tokens = data.get("total_completion_tokens", 0)
                self.total_cached_calls = data.get("total_cached_calls", 0)
                self.total_live_calls = data.get("total_live_calls", 0)
                self.total_usd_spent = data.get("total_usd_spent", 0.0)
                self.model_breakdown = data.get("model_breakdown", {})
        except Exception as e:
            logger.warning(f"Could not load budget state from {self.state_file}: {e}")

    def save_state(self):
        if not self.state_file:
            return
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "max_usd": self.max_usd,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_cached_calls": self.total_cached_calls,
            "total_live_calls": self.total_live_calls,
            "total_usd_spent": round(self.total_usd_spent, 4),
            "model_breakdown": self.model_breakdown,
        }
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        pricing = _pricing(model)
        return (prompt_tokens / 1_000_000.0) * pricing["input"] + (
            completion_tokens / 1_000_000.0
        ) * pricing["output"]

    def preflight(self, model: str, est_prompt_tokens: int, est_completion_tokens: int = 300) -> None:
        """
        Refuse a call *before* it is made if the estimated cost would breach the cap.

        ``check_and_record`` catches an overrun only after the tokens are already
        spent; this stops the run one call earlier, which is what a hard ceiling
        has to mean in practice.
        """
        est = self.calculate_cost(model, est_prompt_tokens, est_completion_tokens)
        if self.total_usd_spent + est > self.max_usd:
            raise BudgetExceededError(
                "Budget ceiling reached before call: spent "
                f"${self.total_usd_spent:.2f}, next call ~${est:.4f}, cap ${self.max_usd:.2f}. "
                "Raise budget.max_usd in the run config, or resume later - the SQLite "
                "cache means completed calls are never paid for twice."
            )

    def remaining_usd(self) -> float:
        return max(self.max_usd - self.total_usd_spent, 0.0)

    def check_and_record(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        is_cached: bool = False,
    ) -> float:
        """
        Records token usage and cost. Raises BudgetExceededError if limit breached.
        """
        if is_cached:
            self.total_cached_calls += 1
            return 0.0

        cost = self.calculate_cost(model, prompt_tokens, completion_tokens)
        if self.total_usd_spent + cost > self.max_usd:
            raise BudgetExceededError(
                f"Budget limit breached: Attempted call cost ${cost:.4f} "
                f"would bring total spend to ${self.total_usd_spent + cost:.2f}, "
                f"exceeding max ceiling of ${self.max_usd:.2f}."
            )

        self.total_live_calls += 1
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_usd_spent += cost

        if model not in self.model_breakdown:
            self.model_breakdown[model] = {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "usd": 0.0,
            }
        self.model_breakdown[model]["calls"] += 1
        self.model_breakdown[model]["prompt_tokens"] += prompt_tokens
        self.model_breakdown[model]["completion_tokens"] += completion_tokens
        self.model_breakdown[model]["usd"] += cost

        self.save_state()
        return cost

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_usd_spent": round(self.total_usd_spent, 4),
            "max_usd": self.max_usd,
            "pct_budget_used": round((self.total_usd_spent / self.max_usd) * 100, 2)
            if self.max_usd > 0
            else 0.0,
            "total_calls": self.total_live_calls + self.total_cached_calls,
            "live_calls": self.total_live_calls,
            "cached_calls": self.total_cached_calls,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "model_breakdown": self.model_breakdown,
        }
