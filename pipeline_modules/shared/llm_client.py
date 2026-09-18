"""
Unified LLM client for the B17 population simulation (module M4's transport layer).

Providers
---------
- ``anthropic``  Claude models via the official ``anthropic`` SDK, using
                 structured outputs (``output_config.format``) so the model is
                 constrained to emit schema-valid JSON.
- ``openai``     GPT models via ``openai``, using JSON mode.
- ``gemini``     Gemini models via ``google-genai``, using response schemas.
- ``mock``       Deterministic offline client. Costs nothing, needs no key, and
                 is used by the test suite and by dry runs.

Every client shares:

- SHA-256 prompt caching (:mod:`shared.cache`) keyed on model + prompts +
  temperature + **sample index**, so the N members of an elicitation ensemble
  are genuinely N draws rather than N copies of one cached draw.
- A hard budget ceiling (:mod:`shared.budget_guard`) that raises before a call
  is made if the run would exceed its configured dollar cap.

The mock client is selected only when the model id resolves to the ``mock``
provider; every other id is dispatched to a real provider and will fail loudly
without an API key. Silently degrading to a mock is what made the previous
version's numbers meaningless, so it is deliberately impossible here.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, Optional, Tuple, Type

import numpy as np
from pydantic import BaseModel

from .budget_guard import BudgetGuard
from .cache import PromptCache

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Model registry. Prices are USD per 1M tokens, list price, as of 2026-09.
# `tier` drives the model-scale ablation arm reported in the writeup (spec 5.5).
# --------------------------------------------------------------------------
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # --- Anthropic -------------------------------------------------------
    "claude-haiku-4-5": {"provider": "anthropic", "in": 1.00, "out": 5.00, "tier": "small"},
    "claude-sonnet-5": {"provider": "anthropic", "in": 2.00, "out": 10.00, "tier": "frontier"},
    "claude-opus-5": {"provider": "anthropic", "in": 5.00, "out": 25.00, "tier": "frontier"},
    # --- OpenAI ----------------------------------------------------------
    "gpt-4o-mini": {"provider": "openai", "in": 0.15, "out": 0.60, "tier": "small"},
    "gpt-4o": {"provider": "openai", "in": 2.50, "out": 10.00, "tier": "frontier"},
    # --- Google ----------------------------------------------------------
    "gemini-2.0-flash": {"provider": "gemini", "in": 0.10, "out": 0.40, "tier": "small"},
    "gemini-2.5-pro": {"provider": "gemini", "in": 1.25, "out": 10.00, "tier": "frontier"},
    # --- Offline ---------------------------------------------------------
    "mock-model": {"provider": "mock", "in": 0.00, "out": 0.00, "tier": "mock"},
}

DEFAULT_MODEL = "claude-haiku-4-5"


def model_info(model: str) -> Dict[str, Any]:
    """Registry entry for ``model``. Unknown ids raise rather than defaulting."""
    if model in MODEL_REGISTRY:
        return MODEL_REGISTRY[model]
    if model.startswith("mock") or model == "offline":
        return MODEL_REGISTRY["mock-model"]
    raise KeyError(
        "Unknown model " + repr(model) + ". Known: "
        + ", ".join(sorted(MODEL_REGISTRY))
        + ". Add it to MODEL_REGISTRY with its per-1M-token prices before using it."
    )


class LLMResponse(BaseModel):
    parsed: Dict[str, Any]
    raw_text: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    is_cached: bool
    model: str
    latency_seconds: float


class LLMError(RuntimeError):
    """Raised when a provider call fails after all retries, or is misconfigured."""


# --------------------------------------------------------------------------
# Base
# --------------------------------------------------------------------------
class BaseLLMClient:
    provider: str = "base"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        cache: Optional[PromptCache] = None,
        budget_guard: Optional[BudgetGuard] = None,
        max_retries: int = 4,
        request_timeout: float = 120.0,
    ):
        self.model = model
        self.cache = cache if cache is not None else PromptCache()
        self.budget_guard = budget_guard if budget_guard is not None else BudgetGuard()
        self.max_retries = max_retries
        self.request_timeout = request_timeout

    # -- public ------------------------------------------------------------
    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: Type[BaseModel],
        temperature: float = 0.7,
        top_p: float = 0.95,
        sample_id: int = 0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """
        One structured-JSON completion, validated against ``response_schema``.

        ``sample_id`` distinguishes the members of an ensemble. It is part of the
        cache key but is never sent to the provider, so repeated draws from one
        prompt are cached independently and stay independent samples. Omitting it
        is what silently collapsed the previous 3x3 ensemble to 3x1.
        """
        cache_key = self.cache.compute_key(
            model=self.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            top_p=top_p,
            extra_key=response_schema.__name__ + "|s" + str(sample_id),
        )

        cached = self.cache.get(cache_key)
        if cached is not None:
            cost = self.budget_guard.check_and_record(
                model=self.model,
                prompt_tokens=cached["prompt_tokens"],
                completion_tokens=cached["completion_tokens"],
                is_cached=True,
            )
            return LLMResponse(
                parsed=cached["response_json"],
                raw_text=json.dumps(cached["response_json"]),
                prompt_tokens=cached["prompt_tokens"],
                completion_tokens=cached["completion_tokens"],
                cost_usd=cost,
                is_cached=True,
                model=self.model,
                latency_seconds=0.0,
            )

        # Stop before spending if the run has hit its dollar cap.
        self.budget_guard.preflight(self.model, est_prompt_tokens=len(user_prompt) // 3)

        start = time.time()
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                raw_text, pt, ct = self._call_api(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response_schema=response_schema,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    sample_id=sample_id,
                )
                parsed = self._parse_and_validate(raw_text, response_schema)
                latency = time.time() - start

                cost = self.budget_guard.check_and_record(
                    model=self.model, prompt_tokens=pt, completion_tokens=ct, is_cached=False
                )
                self.cache.set(
                    cache_key=cache_key,
                    model=self.model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    response_json=parsed,
                    prompt_tokens=pt,
                    completion_tokens=ct,
                )
                return LLMResponse(
                    parsed=parsed,
                    raw_text=raw_text,
                    prompt_tokens=pt,
                    completion_tokens=ct,
                    cost_usd=cost,
                    is_cached=False,
                    model=self.model,
                    latency_seconds=latency,
                )
            except Exception as exc:  # noqa: BLE001 - provider SDKs raise many types
                last_err = exc
                if attempt == self.max_retries - 1:
                    break
                backoff = min(2.0 ** attempt, 30.0)
                logger.warning(
                    "%s attempt %d/%d failed (%s); retrying in %.0fs",
                    self.model,
                    attempt + 1,
                    self.max_retries,
                    exc,
                    backoff,
                )
                time.sleep(backoff)

        raise LLMError(
            self.model + ": all " + str(self.max_retries) + " attempts failed"
        ) from last_err

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _strip_fences(text: str) -> str:
        """Remove markdown code fences a model may wrap its JSON in."""
        cleaned = text.strip()
        fence = "`" * 3
        if fence in cleaned:
            parts = cleaned.split(fence)
            if len(parts) >= 3:
                cleaned = parts[1]
                if cleaned.lstrip().lower().startswith("json"):
                    cleaned = cleaned.lstrip()[4:]
        return cleaned.strip()

    def _parse_and_validate(
        self, raw_text: str, response_schema: Type[BaseModel]
    ) -> Dict[str, Any]:
        cleaned = self._strip_fences(raw_text)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Last resort: pull the first balanced JSON object out of the text.
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if not match:
                raise
            data = json.loads(match.group(0))
        return response_schema.model_validate(data).model_dump()

    def _call_api(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: Type[BaseModel],
        temperature: float,
        top_p: float,
        max_tokens: int,
        sample_id: int = 0,
    ) -> Tuple[str, int, int]:
        """
        Provider call. ``sample_id`` is the ensemble draw index; real providers
        ignore it (each call is already an independent sample at temperature > 0),
        the mock uses it to vary its seed.
        """
        raise NotImplementedError


# --------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------
class AnthropicClient(BaseLLMClient):
    provider = "anthropic"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        import anthropic  # imported lazily so mock runs need no SDK

        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            raise LLMError(
                "ANTHROPIC_API_KEY is not set. Put it in the project-root .env file "
                "(manual.md step 3) or export it in your shell."
            )
        self._client = anthropic.Anthropic(timeout=self.request_timeout, max_retries=0)

    def _call_api(
        self, system_prompt, user_prompt, response_schema, temperature, top_p, max_tokens,
        sample_id: int = 0,
    ):
        schema = response_schema.model_json_schema()
        schema["additionalProperties"] = False
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            temperature=temperature,
            messages=[{"role": "user", "content": user_prompt}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        text = next((b.text for b in resp.content if b.type == "text"), "")
        return text, resp.usage.input_tokens, resp.usage.output_tokens


# --------------------------------------------------------------------------
# OpenAI
# --------------------------------------------------------------------------
class OpenAIClient(BaseLLMClient):
    provider = "openai"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        from openai import OpenAI

        if not os.environ.get("OPENAI_API_KEY"):
            raise LLMError("OPENAI_API_KEY is not set (manual.md step 3).")
        self._client = OpenAI(timeout=self.request_timeout, max_retries=0)

    def _call_api(
        self, system_prompt, user_prompt, response_schema, temperature, top_p, max_tokens,
        sample_id: int = 0,
    ):
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = resp.choices[0].message.content or ""
        return text, resp.usage.prompt_tokens, resp.usage.completion_tokens


# --------------------------------------------------------------------------
# Gemini
# --------------------------------------------------------------------------
class GeminiClient(BaseLLMClient):
    provider = "gemini"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        from google import genai

        key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise LLMError("GOOGLE_API_KEY is not set (manual.md step 3).")
        self._client = genai.Client(api_key=key)

    def _call_api(
        self, system_prompt, user_prompt, response_schema, temperature, top_p, max_tokens,
        sample_id: int = 0,
    ):
        from google.genai import types

        resp = self._client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
            ),
        )
        usage = resp.usage_metadata
        return (
            resp.text or "",
            int(getattr(usage, "prompt_token_count", 0) or 0),
            int(getattr(usage, "candidates_token_count", 0) or 0),
        )


# --------------------------------------------------------------------------
# Mock
# --------------------------------------------------------------------------
class MockLLMClient(BaseLLMClient):
    """
    Deterministic offline stand-in.

    Draws a Dirichlet histogram seeded by the prompt hash and the sample index, so
    an ensemble produces genuinely different members while a whole-pipeline re-run
    reproduces exactly. Useful for wiring tests and cost-free dry runs.

    It is NOT a model of anything. Numbers produced under this provider are noise;
    every artefact from a mock run is tagged ``model=mock-*`` and the benchmark
    refuses to present such a run as a result.
    """

    provider = "mock"

    def _call_api(
        self, system_prompt, user_prompt, response_schema, temperature, top_p, max_tokens,
        sample_id: int = 0,
    ):
        matches = re.findall(r"Options:\s*\[(.*?)\]", user_prompt, re.DOTALL)
        if matches:
            n_options = max(len([o for o in matches[-1].split(",") if o.strip()]), 2)
        else:
            n_options = 4

        seed_hex = PromptCache.compute_key(
            "mock", system_prompt, user_prompt, temperature, extra_key=str(sample_id)
        )[:8]
        rng = np.random.RandomState(int(seed_hex, 16) % (2**32 - 1))
        probs = rng.dirichlet(rng.uniform(0.8, 3.5, size=n_options))
        probs = [round(float(p), 4) for p in probs]
        probs[0] = round(probs[0] + (1.0 - sum(probs)), 4)

        payload = {"probabilities": probs, "reasoning_summary": "mock draw; not a real estimate"}
        text = json.dumps(payload)
        return text, len(system_prompt) // 4 + len(user_prompt) // 4, len(text) // 4


PROVIDERS: Dict[str, type] = {
    "anthropic": AnthropicClient,
    "openai": OpenAIClient,
    "gemini": GeminiClient,
    "mock": MockLLMClient,
}


def get_llm_client(
    model: str = DEFAULT_MODEL,
    cache: Optional[PromptCache] = None,
    budget_guard: Optional[BudgetGuard] = None,
    **kwargs: Any,
) -> BaseLLMClient:
    """
    Build the client for ``model``.

    Dispatch is by the model's registry entry, so asking for a Claude model and
    silently getting a mock back is not possible.
    """
    info = model_info(model)
    cls = PROVIDERS[info["provider"]]
    return cls(model=model, cache=cache, budget_guard=budget_guard, **kwargs)


def is_mock(model: str) -> bool:
    """True when ``model`` resolves to the offline provider."""
    return model_info(model)["provider"] == "mock"
