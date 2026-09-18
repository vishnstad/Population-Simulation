"""Gate 0.

  "a no-op run writes a config snapshot to runs/, the call-count guard kills a
   deliberate overrun, and the provider router fails over cleanly on a
   simulated 429."

All three clauses are asserted here. The 429 is simulated through a fake
transport, so this is a real test of the router rather than a test that needs an
outage to happen.
"""

from __future__ import annotations

import json
import os

import pytest

from popsim.config import load_config, snapshot_run
from popsim.llm.budget import Budget, BudgetExceeded
from popsim.llm.client import (
    AllProvidersExhausted,
    LLMClient,
    LLMResponse,
    ProviderError,
    ProviderSpec,
    QuotaExhausted,
    RateLimited,
)

# ------------------------------------------------------- clause 1: snapshot


def test_noop_run_writes_a_config_snapshot(main_config_path, tmp_path, monkeypatch):
    cfg = load_config(main_config_path)
    monkeypatch.setattr(cfg, "repo_root", tmp_path)
    run_dir = snapshot_run(cfg, note="gate 0 test")

    assert run_dir.is_dir()
    assert (run_dir / "config.snapshot.yaml").exists()
    assert (run_dir / "provenance.json").exists()
    assert (run_dir / f"config.source.{main_config_path.name}").exists()

    prov = json.loads((run_dir / "provenance.json").read_text())
    assert prov["run_id"] == cfg["run_id"]
    assert prov["n_ensemble"] == cfg.n_ensemble

    import yaml
    snap = yaml.safe_load((run_dir / "config.snapshot.yaml").read_text())
    assert snap["partition"]["min_cell"] == 60


def test_overrides_are_recorded_in_the_snapshot(main_config_path, tmp_path, monkeypatch):
    cfg = load_config(main_config_path, overrides={"llm.max_calls_per_run": 7})
    monkeypatch.setattr(cfg, "repo_root", tmp_path)
    run_dir = snapshot_run(cfg)
    prov = json.loads((run_dir / "provenance.json").read_text())
    assert prov["overrides"] == {"llm.max_calls_per_run": 7}
    import yaml
    assert yaml.safe_load((run_dir / "config.snapshot.yaml").read_text())["llm"]["max_calls_per_run"] == 7


# ------------------------------------------------ clause 2: the budget guard


def _spec(name="p1", model="m1", cap=1000):
    return ProviderSpec(name=name, model=model, daily_call_cap=cap, base_url="http://x")


def _ok_transport(counter):
    def t(spec, prompt, *, temperature, max_tokens, json_mode):
        counter.append(spec.name)
        return LLMResponse(text='{"ok": true}', model=spec.model, provider=spec.name,
                           prompt_tokens=10, completion_tokens=5)
    return t


def test_call_ceiling_kills_a_deliberate_overrun(tmp_path):
    sent = []
    budget = Budget(max_calls=3, state_path=tmp_path / "budget.json")
    client = LLMClient([_spec()], cache_dir=tmp_path / "cache", state_dir=tmp_path / "state",
                       budget=budget, default_transport=_ok_transport(sent))

    for i in range(3):
        client.complete(f"prompt {i}", temperature=0.0)
    assert len(sent) == 3

    with pytest.raises(BudgetExceeded, match="call ceiling"):
        client.complete("prompt 4", temperature=0.0)
    assert len(sent) == 3, "the guard must fire BEFORE the call goes out"

    # and it survives a restart: the ceiling is on disk, not in memory
    budget2 = Budget(max_calls=3, state_path=tmp_path / "budget.json")
    assert budget2.state.calls_sent == 3
    with pytest.raises(BudgetExceeded):
        budget2.check()


def test_cache_hits_do_not_consume_budget(tmp_path):
    sent = []
    budget = Budget(max_calls=1, state_path=tmp_path / "budget.json")
    client = LLMClient([_spec()], cache_dir=tmp_path / "cache", state_dir=tmp_path / "state",
                       budget=budget, default_transport=_ok_transport(sent))

    first = client.complete("same prompt", temperature=0.0)
    assert not first.cached
    # Budget is now fully spent, but a replay must still succeed - this is what
    # makes a 37k-call run resumable across weeks of daily quota (0.4c).
    for _ in range(5):
        again = client.complete("same prompt", temperature=0.0)
        assert again.cached and again.text == first.text
    assert len(sent) == 1


def test_repeats_of_the_same_prompt_are_not_collapsed_by_the_cache(tmp_path):
    """n_repeat samples the same prompt on purpose; the cache must not dedupe them."""
    sent = []
    budget = Budget(max_calls=10, state_path=tmp_path / "budget.json")
    client = LLMClient([_spec()], cache_dir=tmp_path / "cache", state_dir=tmp_path / "state",
                       budget=budget, default_transport=_ok_transport(sent))
    for r in range(3):
        client.complete("identical prompt", temperature=0.7, repeat_id=r)
    assert len(sent) == 3, "collapsing repeats would make every ensemble spread zero"


def test_a_free_tier_provider_reporting_cost_stops_the_run(tmp_path):
    def paid(spec, prompt, *, temperature, max_tokens, json_mode):
        return LLMResponse(text="{}", model=spec.model, provider=spec.name, usd=0.01)

    budget = Budget(max_calls=10, cap_usd=0.0, state_path=tmp_path / "budget.json")
    client = LLMClient([_spec()], cache_dir=tmp_path / "cache", state_dir=tmp_path / "state",
                       budget=budget, default_transport=paid)
    client.complete("first", temperature=0.0)
    with pytest.raises(BudgetExceeded, match="paid provider"):
        client.complete("second", temperature=0.0)


# ----------------------------------------------- clause 3: 429 -> clean failover
#
# Gate 0 asks that the router "fails over cleanly on a simulated 429". It does,
# but a 429 is two different events and treating them alike cost a whole run.
#
# On 16 Sep 2026 Gate 3 was launched on Mistral alone. The free tier allows about
# one request a second; the client sent as fast as the network allowed and the
# first call came back
#
#   {"message":"Rate limit exceeded","type":"rate_limited","code":"1300"}
#
# The router read that as the day's allowance being gone, burned Mistral until
# the next day, had nothing to fail over to, and recorded all 960 calls as
# `provider` failures — NO DATA, with `calls_today: 0` in the ledger. Nothing had
# been consumed. So:
#
#   throttle  ("slow down")      -> back off, retry the SAME provider, no burn
#   exhausted ("allowance gone") -> burn until local midnight, fail over
#
# An ambiguous 429 is a throttle, because guessing that way costs retries and
# guessing the other way costs a day.


def test_a_throttling_429_is_retried_on_the_same_provider_and_never_burns_the_day():
    """The 16 Sep regression, as a test."""
    from popsim.llm.client import QuotaExhausted, classify_429

    body = ('mistral 429: {"object":"error","message":"Rate limit exceeded",'
            '"type":"rate_limited","param":null,"code":"1300"}')
    exc = classify_429(body)
    assert not isinstance(exc, QuotaExhausted), (
        "Mistral's per-second throttle must not be read as the day's allowance"
    )


@pytest.mark.parametrize(
    "body, retry_after, exhausted",
    [
        ('{"type":"rate_limited","message":"Rate limit exceeded"}', None, False),
        ("rate limit reached, try again in 2.5s", "3", False),
        ("something opaque", None, False),          # ambiguity -> throttle
        ("You exceeded your current quota", None, True),
        ("requests per day limit reached", None, True),
        ("monthly limit reached for your plan", None, True),
        ("slow down", "7200", True),                # 2 hours is not a throttle
    ],
)
def test_429_classification(body, retry_after, exhausted):
    from popsim.llm.client import QuotaExhausted, classify_429

    assert isinstance(classify_429(body, retry_after), QuotaExhausted) is exhausted


def test_a_throttle_backs_off_and_the_request_still_gets_served(tmp_path):
    seen, slept = [], []

    def transport(spec, prompt, *, temperature, max_tokens, json_mode):
        seen.append(spec.name)
        if spec.name == "primary" and seen.count("primary") == 1:
            raise RateLimited("primary 429: Rate limit exceeded", retry_after_s=2.0)
        return LLMResponse(text='{"ok": 1}', model=spec.model, provider=spec.name,
                           prompt_tokens=8, completion_tokens=4)

    providers = [_spec("primary", "big-model", 1000), _spec("second", "small-model", 500)]
    budget = Budget(max_calls=10, state_path=tmp_path / "budget.json")
    client = LLMClient(providers, cache_dir=tmp_path / "cache", state_dir=tmp_path / "state",
                       budget=budget, default_transport=transport, sleep=slept.append)

    resp = client.complete("hello", temperature=0.0)

    assert resp.provider == "primary", "a throttle must not cost the primary arm"
    assert seen == ["primary", "primary"], "the same provider is retried, not skipped"
    assert 2.0 in slept, "Retry-After must be honoured when the provider sends one"
    assert client.quota["primary"].exhausted_until_day is None, (
        "a throttle must never burn the day — this is the 16 Sep bug"
    )
    assert client.quota["primary"].calls_today == 1


def test_a_throttle_raises_that_provider_pace_so_it_is_not_re_earned(tmp_path):
    def transport(spec, prompt, *, temperature, max_tokens, json_mode):
        if not getattr(transport, "hit", False):
            transport.hit = True
            raise RateLimited("primary 429: Rate limit exceeded")
        return LLMResponse(text="{}", model=spec.model, provider=spec.name)

    providers = [_spec("primary", "big-model", 1000)]
    client = LLMClient(providers, cache_dir=tmp_path / "c", state_dir=tmp_path / "s",
                       budget=Budget(10, state_path=tmp_path / "b.json"),
                       default_transport=transport, sleep=lambda s: None)
    assert client._pace["primary"] == 0.0
    client.complete("hello", temperature=0.0)
    assert client._pace["primary"] > 0.0, (
        "after a throttle the provider must be paced, or every later call re-earns it"
    )


def test_pacing_spaces_consecutive_sends_to_one_provider(tmp_path):
    """Free tiers cap requests per second; without this a run spends its
    attempts earning 429s instead of collecting answers."""
    clock = [100.0]
    slept = []

    def transport(spec, prompt, *, temperature, max_tokens, json_mode):
        return LLMResponse(text="{}", model=spec.model, provider=spec.name)

    def sleep(s):
        slept.append(s)
        clock[0] += s

    spec = _spec("primary", "big-model", 1000)
    spec.min_interval_s = 1.5
    client = LLMClient([spec], cache_dir=tmp_path / "c", state_dir=tmp_path / "s",
                       budget=Budget(10, state_path=tmp_path / "b.json"),
                       default_transport=transport, sleep=sleep,
                       monotonic=lambda: clock[0])

    client.complete("one", temperature=0.0)
    assert slept == [], "the first send waits for nothing"
    clock[0] += 0.4                      # caller did 0.4s of work
    client.complete("two", temperature=0.0)
    assert slept and abs(slept[0] - 1.1) < 1e-6, (
        f"must wait the remaining 1.1s of the 1.5s interval, slept {slept}"
    )
    clock[0] += 99                       # a long gap needs no wait at all
    client.complete("three", temperature=0.0)
    assert len(slept) == 1


def test_an_exhaustion_429_burns_the_day_and_fails_over(tmp_path):
    seen = []

    def transport(spec, prompt, *, temperature, max_tokens, json_mode):
        seen.append(spec.name)
        if spec.name == "primary":
            raise QuotaExhausted("primary 429: You exceeded your current quota")
        return LLMResponse(text='{"ok": 1}', model=spec.model, provider=spec.name,
                           prompt_tokens=8, completion_tokens=4)

    providers = [_spec("primary", "big-model", 1000), _spec("second", "small-model", 500)]
    budget = Budget(max_calls=10, state_path=tmp_path / "budget.json")
    client = LLMClient(providers, cache_dir=tmp_path / "cache", state_dir=tmp_path / "state",
                       budget=budget, default_transport=transport, sleep=lambda s: None)

    resp = client.complete("hello", temperature=0.0)

    assert resp.provider == "second", "the router must serve the request, not surface the 429"
    assert seen == ["primary", "second"], "a real exhaustion is not retried on the same provider"
    assert client.quota["primary"].calls_remaining == 0
    assert client.quota["primary"].exhausted_until_day is not None
    assert client.quota["second"].calls_today == 1
    # Nothing was billed to the run for the refused call.
    assert budget.state.calls_sent == 1
    assert budget.state.per_provider == {"second": 1}

    # A second request skips the burned provider entirely.
    seen.clear()
    client.complete("hello again", temperature=0.0)
    assert seen == ["second"]

    # And the burn survives a restart.
    client2 = LLMClient(providers, cache_dir=tmp_path / "cache2", state_dir=tmp_path / "state",
                        budget=Budget(10, state_path=tmp_path / "b2.json"),
                        default_transport=transport, sleep=lambda s: None)
    assert client2.quota["primary"].calls_remaining == 0
    assert client2.pick_provider().name == "second"


def test_exhausting_every_provider_raises_rather_than_looping(tmp_path):
    def always_429(spec, prompt, *, temperature, max_tokens, json_mode):
        raise RateLimited(f"{spec.name} 429")

    providers = [_spec("a"), _spec("b")]
    client = LLMClient(providers, cache_dir=tmp_path / "cache", state_dir=tmp_path / "state",
                       budget=Budget(10, state_path=tmp_path / "b.json"),
                       default_transport=always_429, sleep=lambda s: None)
    with pytest.raises(AllProvidersExhausted, match="local midnight"):
        client.complete("hi", temperature=0.0)


def test_transient_errors_retry_then_fail_over(tmp_path):
    calls = []

    def flaky(spec, prompt, *, temperature, max_tokens, json_mode):
        calls.append(spec.name)
        if spec.name == "a":
            raise ProviderError("connection reset")
        return LLMResponse(text="{}", model=spec.model, provider=spec.name)

    client = LLMClient([_spec("a"), _spec("b")], cache_dir=tmp_path / "cache",
                       state_dir=tmp_path / "state",
                       budget=Budget(10, state_path=tmp_path / "b.json"),
                       default_transport=flaky, max_retries=2, sleep=lambda s: None)
    resp = client.complete("x", temperature=0.0)
    assert resp.provider == "b"
    assert calls.count("a") == 3, "max_retries=2 means 3 attempts before failing over"
    # A transient error is not a quota event: provider a is still usable tomorrow.
    assert client.quota["a"].exhausted_until_day is None


def test_daily_counter_resets_at_local_midnight(tmp_path):
    from popsim.llm.quota import ProviderQuota

    q = ProviderQuota(name="p", daily_call_cap=2)
    q.record_call(tokens=100)
    q.record_call(tokens=100)
    assert q.calls_remaining == 0 and not q.available

    q.day = "1999-01-01"  # pretend the bucket is yesterday's
    assert q.calls_remaining == 2, "a new local date is a fresh bucket"
    assert q.available


def test_role_order_policy_keeps_the_model_stable(tmp_path):
    sent = []
    providers = [_spec("primary", "big", 1000), _spec("second", "small", 5000)]
    client = LLMClient(providers, cache_dir=tmp_path / "c", state_dir=tmp_path / "s",
                       budget=Budget(10, state_path=tmp_path / "b.json"),
                       policy="role_order", default_transport=_ok_transport(sent))
    client.complete("x", temperature=0.0)
    assert sent == ["primary"], "role_order must not chase the provider with more headroom"

    sent2 = []
    client2 = LLMClient(providers, cache_dir=tmp_path / "c2", state_dir=tmp_path / "s2",
                        budget=Budget(10, state_path=tmp_path / "b2.json"),
                        policy="most_remaining", default_transport=_ok_transport(sent2))
    client2.complete("x", temperature=0.0)
    assert sent2 == ["second"]


# ------------------------------------- the context-window guard (F2 look-alike)


def test_an_oversized_prompt_is_refused_rather_than_truncated(tmp_path):
    """Truncation is silent, and what it removes is the conditioning signal.

    llama.cpp drops the OLDEST tokens when a prompt overruns the window. The
    oldest tokens are the top of the stat card — the cluster definition. Lose
    that and every cluster looks identical to the model, which is exactly the
    output signature of F2. So the client must refuse to send.
    """
    from popsim.llm.client import ContextWindowExceeded

    sent = []
    client = LLMClient([_spec()], cache_dir=tmp_path / "c", state_dir=tmp_path / "s",
                       budget=Budget(10, state_path=tmp_path / "b.json"),
                       num_ctx=2048, default_transport=_ok_transport(sent))

    client.complete("a short card", temperature=0.0, max_tokens=256)
    assert len(sent) == 1

    big = "x" * 8000  # ~2.3k tokens, over a 2048 window
    with pytest.raises(ContextWindowExceeded, match="Nothing was sent"):
        client.complete(big, temperature=0.0, max_tokens=256)
    assert len(sent) == 1, "an oversized prompt must not reach the provider"


def test_the_guard_accounts_for_the_reply_not_just_the_prompt(tmp_path):
    from popsim.llm.client import ContextWindowExceeded

    sent = []
    client = LLMClient([_spec()], cache_dir=tmp_path / "c", state_dir=tmp_path / "s",
                       budget=Budget(10, state_path=tmp_path / "b.json"),
                       num_ctx=2048, default_transport=_ok_transport(sent))
    prompt = "y" * 5000  # ~1.4k tokens: fits alone, not with a large reply
    client.complete(prompt, temperature=0.0, max_tokens=64)
    with pytest.raises(ContextWindowExceeded):
        client.complete(prompt, temperature=0.0, max_tokens=1024)


def test_token_estimate_is_pessimistic():
    from popsim.llm.client import estimate_tokens

    # A guard that fires early costs a config change; one that fires late costs
    # weeks of misread results. So the estimate must not undercount.
    assert estimate_tokens("word " * 100) >= 100
    assert estimate_tokens("") >= 0


# ------------------------------------------------------- .env key loading
#
# `.env.example` says to copy it to `.env`; checklist 0.7 says to export the
# keys; nothing joined the two, so `client.py` read `os.environ` only and every
# hosted-provider call went out with an empty bearer token. The provider returns
# 401, the elicitation loop records it per cell as a `provider` failure — which
# is correct behaviour for a run that must survive a bad afternoon, and
# indistinguishable from the provider being down. A whole run of them on a
# rate-capped free tier costs a day.


def test_dotenv_is_loaded_but_never_overrides_the_shell(tmp_path, monkeypatch):
    from popsim.llm.env import load_dotenv

    (tmp_path / ".env").write_text(
        "# a comment\n"
        "\n"
        "MISTRAL_API_KEY=from_file\n"
        'GROQ_API_KEY="quoted_from_file"\n'
        "export GOOGLE_API_KEY=exported_form\n"
        "NOT_AN_ASSIGNMENT\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MISTRAL_API_KEY", "from_shell")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    loaded = load_dotenv()

    assert os.environ["MISTRAL_API_KEY"] == "from_shell", "the shell must win"
    assert "MISTRAL_API_KEY" not in loaded
    assert os.environ["GROQ_API_KEY"] == "quoted_from_file", "quotes are stripped"
    assert os.environ["GOOGLE_API_KEY"] == "exported_form", "`export ` prefix is handled"
    assert set(loaded) == {"GROQ_API_KEY", "GOOGLE_API_KEY"}


def test_dotenv_is_found_from_a_subdirectory(tmp_path, monkeypatch):
    from popsim.llm.env import find_dotenv

    (tmp_path / ".env").write_text("GROQ_API_KEY=x\n")
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    monkeypatch.chdir(sub)
    found = find_dotenv()
    assert found is not None and found.parent == tmp_path.resolve()


def test_a_missing_dotenv_is_not_an_error(tmp_path, monkeypatch):
    """CI has no .env and must not care."""
    from popsim.llm import env as env_mod

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(env_mod, "_LOADED", {})
    assert env_mod.load_dotenv() == {}


def test_the_load_report_is_stable_across_repeat_calls(tmp_path, monkeypatch):
    """`main()` loads keys before dispatch, so `doctor` calling load_dotenv a
    second time must still be told where they came from — otherwise it reports
    keys read from `.env` as having been set in the shell."""
    from popsim.llm import env as env_mod

    (tmp_path / ".env").write_text("GROQ_API_KEY=x\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(env_mod, "_LOADED", {})
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    first = env_mod.load_dotenv()
    second = env_mod.load_dotenv()
    assert set(first) == {"GROQ_API_KEY"}
    assert second == first, "the second call must report the same provenance"


def test_repeated_throttles_keep_slowing_the_provider_down(tmp_path):
    """The pace doubles from its current value, not from the configured one.

    Doubling the config value would settle at 2x a number already proven too
    fast, and the run would keep earning 429s at a fixed rate for ever.
    """
    def transport(spec, prompt, *, temperature, max_tokens, json_mode):
        raise RateLimited("primary 429: Rate limit exceeded")

    spec = _spec("primary", "big-model", 1000)
    spec.min_interval_s = 1.0
    client = LLMClient([spec], cache_dir=tmp_path / "c", state_dir=tmp_path / "s",
                       budget=Budget(50, state_path=tmp_path / "b.json"),
                       default_transport=transport, sleep=lambda s: None,
                       max_retries=3)
    paces = []
    for i in range(3):
        # Exhausting the only provider is the expected end of each attempt.
        with pytest.raises(AllProvidersExhausted):
            client.complete(f"p{i}", temperature=0.0)
        paces.append(client._pace["primary"])

    assert paces == sorted(paces), f"pace must be monotonic, got {paces}"
    assert paces[-1] > paces[0], "repeated throttles must keep slowing it down"
    assert paces[-1] <= 10.0, "and must stay capped"
    assert client.quota["primary"].exhausted_until_day is None, (
        "no number of throttles may burn the day"
    )
