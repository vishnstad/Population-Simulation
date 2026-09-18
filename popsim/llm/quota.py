"""Per-provider daily counters (checklist 0.4b).

Free tiers are capped **per day, not per dollar**, so the scheduling unit of
this project is "calls remaining today on provider P". That number has to
survive process restarts — a run spans weeks of daily quota — so it lives on
disk and is written through on every call.

Buckets reset at **local midnight**, matching how the providers' own daily
windows are described. The bucket label is the local date; a new local date is a
fresh bucket, which makes reset a lookup rather than a scheduled job.

Token caps (Mistral's ~1B/month) are tracked alongside call caps because a
token-metered provider can be exhausted while its call count looks healthy.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

__all__ = ["ProviderQuota", "QuotaLedger"]


def _today() -> str:
    return _dt.datetime.now().astimezone().strftime("%Y-%m-%d")


def _this_month() -> str:
    return _dt.datetime.now().astimezone().strftime("%Y-%m")


@dataclass
class ProviderQuota:
    name: str
    daily_call_cap: int = 0
    daily_token_cap: int = 0
    monthly_token_cap: int = 0
    day: str = field(default_factory=_today)
    calls_today: int = 0
    tokens_today: int = 0
    month: str = field(default_factory=_this_month)
    tokens_this_month: int = 0
    # Set when a provider 429s; it is skipped until this local date rolls over.
    exhausted_until_day: str | None = None
    last_error: str | None = None

    def _roll(self) -> None:
        today, month = _today(), _this_month()
        if self.day != today:
            self.day = today
            self.calls_today = 0
            self.tokens_today = 0
            if self.exhausted_until_day is not None and self.exhausted_until_day <= today:
                self.exhausted_until_day = None
        if self.month != month:
            self.month = month
            self.tokens_this_month = 0

    @property
    def calls_remaining(self) -> int:
        self._roll()
        if self.exhausted_until_day is not None:
            return 0
        if self.daily_call_cap <= 0:
            return 1_000_000_000  # uncapped (local model)
        return max(0, self.daily_call_cap - self.calls_today)

    @property
    def daily_tokens_remaining(self) -> int:
        """Usually the real constraint, and usually not the one people plan for.

        Groq's free tier is advertised as 1,000 requests/day, but it also caps
        200,000 tokens/day. At this project's ~1.55k tokens per elicitation call
        that is about 129 calls, not 1,000 — an 8x difference between the number
        on the tin and the number that stops the run.
        """
        self._roll()
        if self.daily_token_cap <= 0:
            return 1_000_000_000
        return max(0, self.daily_token_cap - self.tokens_today)

    @property
    def tokens_remaining(self) -> int:
        self._roll()
        if self.monthly_token_cap <= 0:
            return 1_000_000_000
        return max(0, self.monthly_token_cap - self.tokens_this_month)

    @property
    def available(self) -> bool:
        return (
            self.calls_remaining > 0
            and self.tokens_remaining > 0
            and self.daily_tokens_remaining > 0
        )

    def calls_left_today(self, tokens_per_call: int) -> int:
        """How many more calls this provider can actually serve today.

        The minimum of the call cap and what the token caps allow. This is the
        number a run schedule should be built from.
        """
        by_calls = self.calls_remaining
        if tokens_per_call <= 0:
            return by_calls
        by_daily_tokens = self.daily_tokens_remaining // tokens_per_call
        by_monthly_tokens = self.tokens_remaining // tokens_per_call
        return int(min(by_calls, by_daily_tokens, by_monthly_tokens))

    def record_call(self, *, tokens: int = 0) -> None:
        self._roll()
        self.calls_today += 1
        self.tokens_today += int(tokens)
        self.tokens_this_month += int(tokens)

    def mark_exhausted(self, reason: str = "429") -> None:
        """A 429 means today's window is gone; skip until local midnight."""
        self._roll()
        self.exhausted_until_day = (
            _dt.datetime.now().astimezone().date() + _dt.timedelta(days=1)
        ).strftime("%Y-%m-%d")
        self.last_error = reason


    def clear_exhaustion(self) -> bool:
        """Un-burn this provider. Returns whether anything was cleared.

        A burn is meant to be irreversible within the day — that is what stops a
        run rediscovering an exhausted provider every call. It needs an explicit
        way out anyway, because a burn can be *wrong*: until 16 Sep 2026 every
        429 burned the day, including a per-second throttle, and a provider that
        had served nothing sat unusable until local midnight. Never called
        automatically; `popsim quota --clear <provider>` is the only caller.
        """
        had = self.exhausted_until_day is not None
        self.exhausted_until_day = None
        self.last_error = None
        return had


class QuotaLedger:
    """All providers' counters, persisted as one JSON file, written through."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.providers: dict[str, ProviderQuota] = {}
        if self.path.exists():
            # Several sharded elicitation processes share this ledger, and each
            # one writes it by rename. A reader can therefore arrive between the
            # unlink and the rename of someone else's write, or find a partial
            # file on a filesystem that does not make rename atomic. Neither is
            # a reason to fail a run: the ledger is a convenience for pacing, so
            # an unreadable one starts empty and is rebuilt on the next flush.
            try:
                raw = json.loads(self.path.read_text())
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                raw = {}
            self.providers = {k: ProviderQuota(**v) for k, v in raw.items()}

    def register(self, name: str, *, daily_call_cap: int = 0, daily_token_cap: int = 0,
                 monthly_token_cap: int = 0) -> ProviderQuota:
        q = self.providers.get(name)
        if q is None:
            q = ProviderQuota(
                name=name, daily_call_cap=daily_call_cap,
                daily_token_cap=daily_token_cap, monthly_token_cap=monthly_token_cap,
            )
            self.providers[name] = q
        else:
            # Caps come from config and may legitimately change between runs;
            # consumption does not.
            q.daily_call_cap = daily_call_cap
            q.daily_token_cap = daily_token_cap
            q.monthly_token_cap = monthly_token_cap
        q._roll()
        return q

    def __getitem__(self, name: str) -> ProviderQuota:
        return self.providers[name]

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps({k: asdict(v) for k, v in self.providers.items()}, indent=2))
        tmp.replace(self.path)

    def report(self) -> dict[str, dict]:
        return {
            n: {
                "calls_today": q.calls_today,
                "calls_remaining": q.calls_remaining,
                "tokens_today": q.tokens_today,
                "daily_tokens_remaining": q.daily_tokens_remaining,
                "calls_left_today_at_1550_tok": q.calls_left_today(1550),
                "tokens_this_month": q.tokens_this_month,
                "exhausted_until_day": q.exhausted_until_day,
            }
            for n, q in self.providers.items()
        }
