"""Call-count budget guard (checklist 0.4d).

The project runs at ``budget_cap_usd: 0`` on free tiers, so a dollar ceiling
guards nothing. What can still go wrong is a loop bug burning a day's quota in
ten minutes. So the ceiling that matters is **calls per run**, and it is
enforced here, before the call goes out.

A USD ceiling is kept alongside it only as a tripwire: if a provider that was
supposed to be free starts reporting cost, the run stops rather than quietly
spending money.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

__all__ = ["Budget", "BudgetExceeded", "BudgetState"]


class BudgetExceeded(RuntimeError):
    """Raised before a call that would breach the run's ceiling."""


@dataclass
class BudgetState:
    calls_attempted: int = 0
    calls_served_from_cache: int = 0
    calls_sent: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    usd: float = 0.0
    per_provider: dict[str, int] = field(default_factory=dict)


class Budget:
    """Per-run ceiling on network calls.

    Cache hits are counted but do **not** consume budget: the whole point of
    ``cache.py`` being load-bearing (0.4c) is that a resumed run replays for free.
    """

    def __init__(
        self,
        max_calls: int,
        *,
        cap_usd: float = 0.0,
        state_path: Path | None = None,
    ) -> None:
        if max_calls <= 0:
            raise ValueError("max_calls must be positive")
        self.max_calls = int(max_calls)
        self.cap_usd = float(cap_usd)
        self.state_path = Path(state_path) if state_path else None
        self.state = BudgetState()
        if self.state_path and self.state_path.exists():
            self.state = BudgetState(**json.loads(self.state_path.read_text()))

    # ------------------------------------------------------------------ gate
    def check(self, *, n: int = 1) -> None:
        """Raise if sending ``n`` more calls would breach the ceiling."""
        if self.state.calls_sent + n > self.max_calls:
            raise BudgetExceeded(
                f"run call ceiling reached: {self.state.calls_sent} sent, "
                f"max_calls_per_run={self.max_calls}. This guard exists so a loop bug "
                f"cannot burn a day's free-tier quota in ten minutes (checklist 0.4d). "
                f"Raise llm.max_calls_per_run deliberately if the run really is this big."
            )
        if self.cap_usd == 0.0 and self.state.usd > 0.0:
            raise BudgetExceeded(
                f"budget_cap_usd is 0 but ${self.state.usd:.4f} has been reported by a "
                f"provider. A paid provider has leaked into a free-tier run."
            )
        if self.cap_usd > 0.0 and self.state.usd >= self.cap_usd:
            raise BudgetExceeded(f"USD ceiling reached: ${self.state.usd:.4f} >= ${self.cap_usd:.2f}")

    @property
    def remaining(self) -> int:
        return max(0, self.max_calls - self.state.calls_sent)

    # --------------------------------------------------------------- record
    def record_attempt(self) -> None:
        self.state.calls_attempted += 1

    def record_cache_hit(self) -> None:
        self.state.calls_served_from_cache += 1
        self._persist()

    def record_sent(
        self,
        provider: str,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        usd: float = 0.0,
    ) -> None:
        self.state.calls_sent += 1
        self.state.prompt_tokens += int(prompt_tokens)
        self.state.completion_tokens += int(completion_tokens)
        self.state.usd += float(usd)
        self.state.per_provider[provider] = self.state.per_provider.get(provider, 0) + 1
        self._persist()

    def _persist(self) -> None:
        if not self.state_path:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self.state), indent=2))
        tmp.replace(self.state_path)

    def summary(self) -> dict:
        d = asdict(self.state)
        d["max_calls"] = self.max_calls
        d["remaining"] = self.remaining
        return d
