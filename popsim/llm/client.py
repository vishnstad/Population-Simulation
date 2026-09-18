"""LLM client with provider router (checklist 0.4, 0.4a-0.4d).

Everything that talks to a model goes through :class:`LLMClient`. It composes
four things that the checklist insists on having before the first call is made:

* :mod:`popsim.llm.cache`  — every completed call on disk, so a resume is free
* :mod:`popsim.llm.quota`  — per-provider daily counters, reset at local midnight
* :mod:`popsim.llm.budget` — a per-run call ceiling, checked *before* sending
* the router below        — provider chosen by remaining quota, 429 falls through

The ordering inside :meth:`LLMClient.complete` is deliberate:

    cache lookup -> budget check -> router pick -> send -> record -> cache write

Cache first, because a cache hit must cost neither budget nor quota (that is
what makes a three-week resumable run possible). Budget before router, because
the guard against a loop bug has to fire even when every provider is healthy.
Cache write last and unconditional, because a response that was paid for in
quota and then lost is the expensive failure this module exists to prevent.

Transport
---------
Providers are pluggable ``Transport`` callables so that the whole router can be
tested without a network — Gate 0 requires simulating a 429 and watching the
failover, which is a test, not an outage. Real transports are thin: an OpenAI
-compatible chat/completions POST covers Mistral, Groq, Cerebras and Ollama; the
Google arm differs enough to get its own.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .budget import Budget
from .cache import CachedResponse, CacheKey, ResponseCache
from .quota import QuotaLedger

__all__ = [
    "AllProvidersExhausted",
    "LLMClient",
    "LLMResponse",
    "ProviderError",
    "ProviderSpec",
    "QuotaExhausted",
    "RateLimited",
    "classify_429",
    "http_transport",
]

log = logging.getLogger("popsim.llm")


class ProviderError(RuntimeError):
    """A provider failed in a way that is worth trying elsewhere."""


class RateLimited(ProviderError):
    """HTTP 429 that means *slow down*, not *come back tomorrow*.

    These are two different things and 0.4a conflated them. The checklist says
    the free tiers are "rate-capped per day, not per dollar", which is true and
    incomplete: they are rate-capped per *second* as well. Mistral's free tier
    allows roughly one request a second, and the client fires as fast as the
    network allows, so the very first calls of a run draw

        {"message":"Rate limit exceeded","type":"rate_limited","code":"1300"}

    On 16 Sep 2026 one of those burned Mistral until the next day with
    ``calls_today: 0``, and since it was the only active provider the whole
    960-call Gate 3 run recorded `provider` failures and returned NO DATA.
    Nothing had been consumed; the ledger simply believed it had.

    So a throttle backs off and retries the *same* provider, honouring
    ``Retry-After`` when the provider sends one, and only :class:`QuotaExhausted`
    burns the day.
    """

    def __init__(self, message: str, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class QuotaExhausted(RateLimited):
    """The provider's allowance really is gone. Burn it and fail over.

    Distinguished from a throttle by what the provider says — an explicit
    daily/monthly quota or credit refusal — never by the bare 429 status, and
    never assumed. Guessing this way round costs a day; guessing the other way
    costs a handful of retries, so an ambiguous 429 is treated as a throttle.
    """


class AllProvidersExhausted(RuntimeError):
    """Every configured provider is out of quota or failing."""


class ContextWindowExceeded(RuntimeError):
    """The prompt does not fit in the model's context window.

    This is its own exception, and it is fatal rather than retried, because the
    way a too-long prompt fails is worse than an error: Ollama (and llama.cpp
    generally) **silently drops the oldest tokens** and answers on what is left.
    The oldest tokens are the top of the stat card — the cluster definition,
    which is the entire conditioning signal. Every cluster then looks identical
    to the model and every cluster gets the same answer.

    That is indistinguishable, in the output, from F2 (between-cluster collapse)
    — the failure the spec calls the most likely way this project silently fails
    while looking like it works. You would run the Gate 3 permutation test, see
    permuted ≈ real, and conclude the premise is dead, when the actual cause was
    a context-window setting. So: refuse to send, loudly.
    """


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    usd: float = 0.0
    cached: bool = False
    attempts: int = 1
    finish_reason: str | None = None


@dataclass
class ProviderSpec:
    name: str
    model: str
    daily_call_cap: int = 0
    daily_token_cap: int = 0
    monthly_token_cap: int = 0
    role: str = ""
    base_url: str = ""
    api_key_env: str = ""
    #: Minimum seconds between sends to this provider. Free tiers cap requests
    #: per second as well as per day, and a run that ignores that spends its
    #: attempts on 429s instead of answers. 0 means no pacing (local models).
    min_interval_s: float = 0.0


#: Words a provider uses when the *allowance* is gone rather than the rate. Kept
#: narrow and matched against the response body: the failure that matters is
#: reading a throttle as exhaustion, so anything not clearly about a quota,
#: credit or plan limit stays a throttle.
#: Default seconds between sends, per provider. Free tiers throttle per second
#: and publish the number only in their consoles, so these are conservative
#: starting points that a config can override; the client also raises a
#: provider's pace on its own the first time that provider throttles us.
#: Mistral's free tier is about one request per second, which is what produced
#: the 16 Sep NO DATA run.
_DEFAULT_MIN_INTERVAL_S: dict[str, float] = {
    "mistral": 1.1,
    "groq": 2.2,        # 30 req/min on the free tier
    "google": 4.1,      # 15 req/min on Flash free
    "cerebras": 2.2,
    "ollama": 0.0,      # local, unlimited
    "ollama_modern": 0.0,
}

_EXHAUSTION_MARKERS = (
    "quota", "credit", "insufficient_quota", "monthly limit", "daily limit",
    "usage limit", "billing", "plan limit", "exceeded your current",
    "tokens per day", "requests per day", "spending limit",
)


def classify_429(body: str, retry_after: str | None = None) -> RateLimited:
    """Turn a 429 into a throttle or an exhaustion, on evidence.

    ``Retry-After`` is the strongest signal when present: a provider telling you
    to come back in seconds is throttling, and one telling you to come back in
    hours has closed the window. Failing that, the body is matched against
    :data:`_EXHAUSTION_MARKERS`. **Ambiguity resolves to a throttle**, because
    mistaking a throttle for exhaustion costs a whole day of a rate-capped tier
    and the reverse costs a few retries.
    """
    wait: float | None = None
    if retry_after:
        try:
            wait = max(0.0, float(str(retry_after).strip()))
        except (TypeError, ValueError):
            wait = None

    low = (body or "").lower()
    exhausted = any(m in low for m in _EXHAUSTION_MARKERS)
    # Over ten minutes is not a per-second throttle whatever the body says.
    if wait is not None and wait > 600:
        exhausted = True

    if exhausted:
        return QuotaExhausted(body or "429", retry_after_s=wait)
    return RateLimited(body or "429", retry_after_s=wait)


class Transport(Protocol):
    def __call__(self, spec: ProviderSpec, prompt: str, *, temperature: float,
                 max_tokens: int, json_mode: bool) -> LLMResponse: ...


def estimate_tokens(text: str) -> int:
    """Rough token count. Deliberately pessimistic.

    ~4 characters per token is the usual English rule of thumb; stat cards are
    denser than prose (numbers, punctuation, option labels) so this uses 3.5 and
    rounds up. A guard that fires slightly early costs a config change; a guard
    that fires slightly late costs weeks of misread results.
    """
    return int(len(text) / 3.5) + 1


# --------------------------------------------------------------------- client
class LLMClient:
    """Cache-first, quota-aware, resumable LLM caller."""

    #: How a provider is picked among those with quota left.
    #:
    #: ``role_order``    — first available in configured order. Keeps the model
    #:                     identity of a run stable, which matters because a run
    #:                     that silently switches models has no interpretable
    #:                     model-tier story (checklist 7.3a).
    #: ``most_remaining`` — the provider with the most headroom. Maximises
    #:                     throughput; mixes models within a run.
    POLICIES = ("role_order", "most_remaining")

    def __init__(
        self,
        providers: Iterable[ProviderSpec],
        *,
        cache_dir: str | os.PathLike[str],
        state_dir: str | os.PathLike[str],
        budget: Budget,
        policy: str = "role_order",
        transports: dict[str, Transport] | None = None,
        default_transport: Transport | None = None,
        max_retries: int = 2,
        retry_backoff_s: float = 1.0,
        monotonic: Callable[[], float] = time.monotonic,
        num_ctx: int = 0,
        ctx_safety: float = 0.9,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.providers = list(providers)
        if not self.providers:
            raise ValueError("no providers configured")
        if policy not in self.POLICIES:
            raise ValueError(f"policy must be one of {self.POLICIES}, got {policy!r}")
        self.policy = policy
        self.cache = ResponseCache(cache_dir)
        self.budget = budget
        self.max_retries = int(max_retries)
        self.num_ctx = int(num_ctx)
        self.ctx_safety = float(ctx_safety)
        self.retry_backoff_s = float(retry_backoff_s)
        self._sleep = sleep
        # Per-provider minimum seconds between sends, and when each last sent.
        # Free tiers cap requests per second as well as per day; without this the
        # run spends its attempts earning 429s instead of collecting answers.
        # Seeded from config and raised whenever a provider throttles us.
        #: Consecutive successes since a provider was last throttled. The pace
        #: floor decays back toward the configured interval after enough of them
        #: — see `_note_success`.
        self._ok_streak: dict[str, int] = {}
        self._pace: dict[str, float] = {
            sp.name: float(sp.min_interval_s) for sp in self.providers
        }
        self._last_send: dict[str, float] = {}
        self._now = monotonic

        state = Path(state_dir)
        state.mkdir(parents=True, exist_ok=True)
        self.quota = QuotaLedger(state / "quota.json")
        for spec in self.providers:
            self.quota.register(
                spec.name,
                daily_call_cap=spec.daily_call_cap,
                daily_token_cap=spec.daily_token_cap,
                monthly_token_cap=spec.monthly_token_cap,
            )
        self.quota.flush()

        self.transports = dict(transports or {})
        self.default_transport = default_transport or http_transport

    # ------------------------------------------------------------ router
    def available_providers(self) -> list[ProviderSpec]:
        return [s for s in self.providers if self.quota[s.name].available]

    def pick_provider(self, *, skip: set[str] | None = None) -> ProviderSpec:
        skip = skip or set()
        candidates = [s for s in self.available_providers() if s.name not in skip]
        if not candidates:
            raise AllProvidersExhausted(
                "no provider has quota left today. Ledger: "
                f"{json.dumps(self.quota.report())}. Free-tier windows reset at local "
                "midnight; the run is resumable, so stopping here loses nothing."
            )
        if self.policy == "most_remaining":
            # Rank by calls actually servable today, which the token caps often
            # decide, not by the advertised request cap.
            return max(candidates, key=lambda s: self.quota[s.name].calls_left_today(1550))
        return candidates[0]

    # ------------------------------------------------------------ main call
    def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        paraphrase_id: int = 0,
        repeat_id: int = 0,
        json_mode: bool = True,
        num_ctx: int | None = None,
    ) -> LLMResponse:
        """Return a completion, from cache if this exact draw was made before.

        ``model`` pins the cache key. When None it resolves to the model of the
        provider the router would pick *first*, so that a cache built on a run's
        primary provider is still hit on resume.
        """
        self.budget.record_attempt()

        ctx = num_ctx if num_ctx is not None else self.num_ctx
        if ctx:
            need = estimate_tokens(prompt) + max_tokens
            if need > ctx * self.ctx_safety:
                raise ContextWindowExceeded(
                    f"prompt ~{estimate_tokens(prompt)} tokens + {max_tokens} reserved for the "
                    f"reply = ~{need}, against a context window of {ctx} "
                    f"(guard trips at {self.ctx_safety:.0%}). Nothing was sent.\n"
                    f"  Raise llm.num_ctx, or shrink the card via "
                    f"elicitation.anchors_per_card.\n"
                    f"  For Ollama the window is baked into the model, not the request:\n"
                    f"    printf 'FROM <base>\\nPARAMETER num_ctx {max(8192, ctx)}\\n' > Modelfile\n"
                    f"    ollama create <name>-ctx -f Modelfile"
                )

        resolved_model = model or self._cache_model()
        key = CacheKey(
            model=resolved_model,
            prompt=prompt,
            temperature=temperature,
            paraphrase_id=paraphrase_id,
            repeat_id=repeat_id,
        )
        hit = self.cache.get(key)
        if hit is not None:
            self.budget.record_cache_hit()
            return LLMResponse(
                text=hit.text, model=hit.model, provider=hit.provider,
                prompt_tokens=hit.prompt_tokens, completion_tokens=hit.completion_tokens,
                usd=hit.usd, cached=True, attempts=hit.attempts,
                finish_reason=hit.finish_reason,
            )

        # Guard fires even when every provider is healthy: this is the loop-bug
        # tripwire, not a quota check.
        self.budget.check()

        skip: set[str] = set()
        last_exc: Exception | None = None
        attempts = 0

        while True:
            try:
                spec = self.pick_provider(skip=skip)
            except AllProvidersExhausted as exc:
                # Surface *why* the last provider gave up, not just that nothing
                # is left: "all exhausted" with no cause is the least actionable
                # message a three-week run can end on.
                raise AllProvidersExhausted(
                    f"{exc} Last provider error: {last_exc!r}"
                ) from last_exc
            transport = self.transports.get(spec.name, self.default_transport)
            for _try in range(self.max_retries + 1):
                attempts += 1
                self._wait_for_pace(spec.name)
                try:
                    resp = transport(
                        spec, prompt, temperature=temperature,
                        max_tokens=max_tokens, json_mode=json_mode,
                    )
                except QuotaExhausted as exc:
                    # The allowance really is gone. Burn it until local midnight,
                    # do not count it against the run budget - nothing was served.
                    log.warning(
                        "%s reports its allowance exhausted; burning it for today "
                        "and failing over: %s", spec.name, exc,
                    )
                    self.quota[spec.name].mark_exhausted(str(exc))
                    self.quota.flush()
                    skip.add(spec.name)
                    last_exc = exc
                    break
                except RateLimited as exc:
                    # A throttle, not the daily window. Back off on the SAME
                    # provider: burning it here is what turned one Mistral 429
                    # into a 960-call run of NO DATA with calls_today at 0.
                    last_exc = exc
                    if _try < self.max_retries:
                        wait = exc.retry_after_s
                        if wait is None:
                            wait = self.retry_backoff_s * (2 ** _try)
                        # A throttle is about pace, so raise the pace floor too,
                        # or every later call re-earns the same 429. Doubles the
                        # CURRENT pace, not the configured one, so a provider
                        # that keeps throttling keeps slowing down instead of
                        # settling at twice a value that was already too fast.
                        current = self._pace.get(spec.name, 0.0)
                        self._pace[spec.name] = min(
                            max(current, max(spec.min_interval_s, 0.25)) * 2, 10.0
                        )
                        log.info(
                            "%s throttled; waiting %.1fs and retrying it "
                            "(pace now %.2fs/call)",
                            spec.name, wait, self._pace[spec.name],
                        )
                        self._sleep(wait)
                        continue
                    log.warning(
                        "%s still throttled after %d attempts; failing over "
                        "without burning its day", spec.name, attempts,
                    )
                    skip.add(spec.name)
                    break
                except ProviderError as exc:
                    last_exc = exc
                    if _try < self.max_retries:
                        self._sleep(self.retry_backoff_s * (2 ** _try))
                        continue
                    log.warning("%s failed %d times: %s; failing over", spec.name, attempts, exc)
                    skip.add(spec.name)
                    break
                else:
                    resp.attempts = attempts
                    self._note_success(spec)
                    self.quota[spec.name].record_call(
                        tokens=resp.prompt_tokens + resp.completion_tokens
                    )
                    self.quota.flush()
                    self.budget.record_sent(
                        spec.name,
                        prompt_tokens=resp.prompt_tokens,
                        completion_tokens=resp.completion_tokens,
                        usd=resp.usd,
                    )
                    # Checkpoint after every single call (0.4c). Written under the
                    # key that was looked up, so the resolved model stays stable
                    # even when the router failed over to a different provider.
                    self.cache.put(key, CachedResponse(
                        text=resp.text, model=key.model, provider=resp.provider,
                        prompt_tokens=resp.prompt_tokens,
                        completion_tokens=resp.completion_tokens,
                        usd=resp.usd, finish_reason=resp.finish_reason,
                        attempts=attempts,
                        meta={"served_by_model": resp.model, "router_policy": self.policy},
                    ))
                    return resp
            # fell out of the retry loop -> try the next provider

    def _cache_model(self) -> str:
        try:
            return self.pick_provider().model
        except AllProvidersExhausted:
            return self.providers[0].model

    #: Successes required before the pace floor is relaxed one step, and the
    #: factor it relaxes by. Slow to recover on purpose: a provider that is
    #: genuinely at its cap will throttle again long before the floor is back.
    PACE_RECOVERY_AFTER = 40
    PACE_RECOVERY_FACTOR = 0.7

    def _note_success(self, spec: ProviderSpec) -> None:
        """Let a throttled provider's pace decay back toward its configured one.

        The back-off doubles the pace floor on every 429 and used to never come
        back down, which is right for a minute and wrong for a day: an overnight
        run that met one bad patch early spent the remaining eleven hours at
        10 s/call. Observed on the 14B arm — a run pacing at 28 calls/min dropped
        to 6 and stayed there, and the only cure was restarting the process.

        So: after a streak of clean sends, step the floor back down, never below
        the configured `min_interval_s`. Recovery is deliberately slower than the
        back-off, so a provider actually at its cap re-throttles before the floor
        returns rather than oscillating.
        """
        name = spec.name
        floor = max(float(spec.min_interval_s), 0.0)
        cur = self._pace.get(name, floor)
        if cur <= floor:
            self._ok_streak[name] = 0
            return
        n = self._ok_streak.get(name, 0) + 1
        if n < self.PACE_RECOVERY_AFTER:
            self._ok_streak[name] = n
            return
        self._ok_streak[name] = 0
        self._pace[name] = max(floor, cur * self.PACE_RECOVERY_FACTOR)
        log.info("%s: %d clean sends, pace floor relaxed to %.2fs/call",
                 name, self.PACE_RECOVERY_AFTER, self._pace[name])

    # ------------------------------------------------------------ reporting
    def _wait_for_pace(self, provider: str) -> None:
        """Sleep so consecutive sends to one provider are min_interval_s apart.

        Sleeps before the send rather than after, so a provider paced at 1 s/call
        is not slowed by work the caller does in between, and so a resumed run
        that hits the cache pays nothing here at all (the cache is checked before
        the router ever picks a provider).
        """
        gap = self._pace.get(provider, 0.0)
        if gap <= 0:
            return
        last = self._last_send.get(provider)
        now = self._now()
        if last is not None:
            wait = gap - (now - last)
            if wait > 0:
                self._sleep(wait)
                now = self._now()
        self._last_send[provider] = now

    def report(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "cache": self.cache.stats(),
            "cache_entries": len(self.cache),
            "budget": self.budget.summary(),
            "quota": self.quota.report(),
        }

    # ------------------------------------------------------------ factory
    @classmethod
    def from_config(cls, cfg, *, run_dir: Path | None = None, **kw) -> LLMClient:
        """Build a client from a :class:`popsim.config.Config`."""
        active = set(cfg.get("llm.active_providers") or [])
        specs = [
            ProviderSpec(
                name=p["name"],
                model=p["model"],
                daily_call_cap=int(p.get("daily_call_cap", 0)),
                daily_token_cap=int(p.get("daily_token_cap", 0)),
                monthly_token_cap=int(p.get("monthly_token_cap", 0)),
                role=p.get("role", ""),
                base_url=p.get("base_url", _family_default(p["name"], _DEFAULT_BASE_URLS)),
                api_key_env=p.get("api_key_env",
                                  _family_default(p["name"], _DEFAULT_KEY_ENVS)),
                min_interval_s=float(
                    p.get("min_interval_s",
                          _family_default(p["name"], _DEFAULT_MIN_INTERVAL_S, 0.0))
                ),
            )
            for p in cfg["llm.providers"]
            if not active or p["name"] in active
        ]
        root = cfg.repo_root
        budget = Budget(
            int(cfg["llm.max_calls_per_run"]),
            cap_usd=float(cfg["llm.budget_cap_usd"]),
            state_path=(run_dir / "budget.json") if run_dir else None,
        )
        return cls(
            specs,
            cache_dir=root / cfg.get("llm.cache_dir", ".llm_cache"),
            state_dir=root / cfg.get("llm.state_dir", ".llm_state"),
            budget=budget,
            max_retries=int(cfg.get("elicitation.max_retries", 2)),
            num_ctx=int(cfg.get("llm.num_ctx", 0)),
            **kw,
        )


def _family_default(name: str, table: dict, fallback=""):
    """Look a provider's defaults up by its family.

    Several arms of the same provider run side by side — ``mistral`` for the
    14B, ``mistral_8b`` for the throughput arm, ``mistral_3b`` for the bottom of
    the §7.3a tier curve — and they need separate names because the quota ledger,
    the pacing and the per-model capacity are all per name. Mistral allocates
    req/min per *model*, so one entry per name is the only thing that can pace
    them correctly. The endpoint and the key are per family, so they come from
    the part of the name before the first underscore.
    """
    if name in table:
        return table[name]
    return table.get(name.split("_", 1)[0], fallback)


_DEFAULT_BASE_URLS = {
    "ollama": "http://localhost:11434/v1",
    "mistral": "https://api.mistral.ai/v1",
    "groq": "https://api.groq.com/openai/v1",
    "cerebras": "https://api.cerebras.ai/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai",
}
_DEFAULT_KEY_ENVS = {
    "mistral": "MISTRAL_API_KEY",
    "groq": "GROQ_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "google": "GOOGLE_API_KEY",
    "ollama": "",
}


# ------------------------------------------------------------------ transport
def http_transport(
    spec: ProviderSpec, prompt: str, *, temperature: float, max_tokens: int, json_mode: bool
) -> LLMResponse:
    """OpenAI-compatible chat/completions POST.

    Covers Mistral, Groq, Cerebras, Google's OpenAI-compat endpoint and a local
    Ollama. Deliberately stdlib-only: one less pin to keep in step with 0.3.
    """
    if not spec.base_url:
        raise ProviderError(f"provider {spec.name!r} has no base_url")
    api_key = os.environ.get(spec.api_key_env, "") if spec.api_key_env else ""
    if spec.api_key_env and not api_key:
        raise ProviderError(
            f"{spec.name}: ${spec.api_key_env} is not set. Register the free tier and "
            f"export the key (checklist 0.7)."
        )

    body = {
        "model": spec.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    req = urllib.request.Request(
        f"{spec.base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            payload = json.loads(r.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:400].decode("utf-8", "replace")
        if exc.code == 429:
            # The rate-limit headers are the only place a provider says what
            # the limit actually *is*. Mistral documents its limits as
            # workspace-level and console-only, so on a 429 these headers are
            # the machine-readable answer to "per-second, or no capacity at
            # all?" — a distinction that decides whether pacing can help.
            hdrs = dict(exc.headers or {})
            rl = {
                k: v for k, v in hdrs.items()
                if k.lower().startswith(("x-ratelimit", "ratelimit"))
                or k.lower() in ("retry-after", "x-request-id")
            }
            raise classify_429(
                f"{spec.name} 429: {detail}"
                + (f" | headers: {json.dumps(rl, sort_keys=True)}" if rl else
                   " | no rate-limit headers sent"),
                hdrs.get("Retry-After"),
            ) from exc
        raise ProviderError(f"{spec.name} HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise ProviderError(f"{spec.name} transport failure: {exc}") from exc

    try:
        choice = payload["choices"][0]
        text = choice["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise ProviderError(f"{spec.name}: unexpected response shape: {str(payload)[:300]}") from exc
    usage = payload.get("usage") or {}
    return LLMResponse(
        text=text,
        model=payload.get("model", spec.model),
        provider=spec.name,
        prompt_tokens=int(usage.get("prompt_tokens", 0)),
        completion_tokens=int(usage.get("completion_tokens", 0)),
        usd=0.0,  # free tiers; budget.check() trips if a provider ever reports cost
        finish_reason=choice.get("finish_reason"),
    )
