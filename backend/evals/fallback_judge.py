# evals/fallback_judge.py
"""
DeepEval judge that falls back across multiple Groq models.

Groq free-tier limits are applied PER MODEL, so when one model returns a 429,
the same request is retried immediately on the next model. If every model is
cooling down, the judge sleeps until the earliest one recovers (up to
max_total_wait seconds), then raises.
"""
import asyncio
import json
import os
import re
import time
from typing import Optional

from deepeval.models import DeepEvalBaseLLM
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel

GROQ_URL = "https://api.groq.com/openai/v1"

# Priority order: strongest judge first.
# (model_id, extra request params sent via extra_body)
# Skipped on purpose: prompt-guard-* (classifiers, not chat models) and
# gpt-oss-safeguard-20b (safety classifier, only 3 RPM / 2K TPM).
GROQ_MODELS = [
    ("openai/gpt-oss-120b", {"reasoning_effort": "low"}),
    ("qwen/qwen3.8-27b", {})
]


class Provider:
    def __init__(self, model: str, api_key: str, extra: Optional[dict] = None,
                 json_mode: bool = True, timeout: float = 40):
        self.name = model
        self.model = model
        self.extra = extra or {}
        self.json_mode = json_mode
        self.cooldown_until = 0.0
        self.client = OpenAI(base_url=GROQ_URL, api_key=api_key, max_retries=0, timeout=timeout)
        self.aclient = AsyncOpenAI(base_url=GROQ_URL, api_key=api_key, max_retries=0, timeout=timeout)

    def available(self) -> bool:
        return time.monotonic() >= self.cooldown_until

    def cool(self, seconds: float):
        self.cooldown_until = time.monotonic() + seconds


def default_providers() -> list[Provider]:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY missing from environment")
    return [Provider(m, key, extra) for m, extra in GROQ_MODELS]


class _CallState:
    """Per-request bookkeeping so one bad request can't loop forever."""

    def __init__(self, max_total_wait: float, max_attempts: int):
        self.deadline = time.monotonic() + max_total_wait
        self.max_attempts = max_attempts
        self.attempts = 0
        self.rejected: set[str] = set()  # models that rejected THIS request (e.g. 413)


class FallbackJudge(DeepEvalBaseLLM):
    def __init__(self, providers: Optional[list[Provider]] = None,
                 max_total_wait: float = 600, max_attempts: int = 15):
        self.providers = providers or default_providers()
        self.max_total_wait = max_total_wait
        self.max_attempts = max_attempts
        self.last_model: Optional[str] = None  # which model answered most recently

    # ------------------------------------------------------------------ DeepEval interface
    def load_model(self):
        return self

    def get_model_name(self) -> str:
        return "fallback-judge(" + ", ".join(p.name for p in self.providers) + ")"

    def generate(self, prompt: str, schema: Optional[type[BaseModel]] = None):
        state = _CallState(self.max_total_wait, self.max_attempts)
        while True:
            candidates = self._candidates(state)
            if not candidates:
                time.sleep(self._wait_time(state))
                continue
            for p in candidates:
                t = time.monotonic()
                try:
                    resp = p.client.chat.completions.create(**self._kwargs(p, prompt, schema))
                    out = self._parse(resp.choices[0].message.content, schema)
                    self._log_ok(p, t)
                    return out
                except Exception as e:
                    self._handle(p, e, state)

    async def a_generate(self, prompt: str, schema: Optional[type[BaseModel]] = None):
        state = _CallState(self.max_total_wait, self.max_attempts)
        while True:
            candidates = self._candidates(state)
            if not candidates:
                await asyncio.sleep(self._wait_time(state))
                continue
            for p in candidates:
                t = time.monotonic()
                try:
                    resp = await p.aclient.chat.completions.create(**self._kwargs(p, prompt, schema))
                    out = self._parse(resp.choices[0].message.content, schema)
                    self._log_ok(p, t)
                    return out
                except Exception as e:
                    self._handle(p, e, state)

    # ------------------------------------------------------------------ internals
    def _kwargs(self, p: Provider, prompt: str, schema):
        if schema is not None:
            prompt += (
                "\n\nRespond with ONLY a valid JSON object matching this JSON schema. "
                "No markdown, no commentary:\n" + json.dumps(schema.model_json_schema())
            )
        kw = dict(
            model=p.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        if schema is not None and p.json_mode:
            kw["response_format"] = {"type": "json_object"}
        if p.extra:
            kw["extra_body"] = p.extra
        return kw

    @staticmethod
    def _parse(text: Optional[str], schema):
        text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
        if schema is None:
            return text
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("no JSON object in model output")
        return schema(**json.loads(text[start:end + 1]))  # ValueError/ValidationError -> next model

    def _candidates(self, state: _CallState) -> list[Provider]:
        if state.attempts >= state.max_attempts:
            raise RuntimeError(f"Gave up after {state.attempts} attempts on one request")
        live = [p for p in self.providers if p.name not in state.rejected]
        if not live:
            raise RuntimeError(
                "Request rejected by every model (likely too large for the per-minute "
                "token cap). Reduce top_k / chunk size."
            )
        return [p for p in live if p.available()]

    def _wait_time(self, state: _CallState) -> float:
        live = [p for p in self.providers if p.name not in state.rejected]
        soonest = min(p.cooldown_until for p in live)
        wait = max(soonest - time.monotonic(), 0) + 0.5
        if time.monotonic() + wait > state.deadline:
            raise RuntimeError(
                "All Groq models are rate-limited beyond max_total_wait "
                "(daily token cap probably exhausted)."
            )
        print(f"[judge] all models cooling, sleeping {wait:.0f}s")
        return wait

    def _log_ok(self, p: Provider, started: float):
        self.last_model = p.name
        print(f"[judge] {p.name} ok in {time.monotonic() - started:.1f}s")

    @staticmethod
    def _retry_after(e: RateLimitError) -> float:
        try:
            return min(float(e.response.headers.get("retry-after", 20)), 120.0)
        except Exception:
            return 20.0

    def _handle(self, p: Provider, e: Exception, state: _CallState):
        state.attempts += 1
        if isinstance(e, RateLimitError):
            secs = self._retry_after(e)
            print(f"[judge] {p.name} rate-limited, cooling {secs:.0f}s -> next model")
            p.cool(secs)
        elif isinstance(e, APIStatusError) and e.status_code == 413:
            print(f"[judge] {p.name} says request too large -> next model")
            state.rejected.add(p.name)  # only for this request, no global cooldown
        elif isinstance(e, APIStatusError) and e.status_code in (401, 403):
            raise RuntimeError(f"Groq auth error ({e.status_code}): check GROQ_API_KEY") from e
        elif isinstance(e, APIStatusError) and e.status_code in (400, 404):
            print(f"[judge] {p.name} bad request/model ({e.status_code}): "
                  f"{str(e)[:120]} -> disabling for 1h")
            p.cool(3600)
        elif isinstance(e, (APIStatusError, APIConnectionError, APITimeoutError)):
            print(f"[judge] {p.name} transient error: {repr(e)[:120]} -> next model")
            p.cool(15)
        elif isinstance(e, ValueError):  # bad JSON / pydantic validation
            print(f"[judge] {p.name} returned invalid JSON -> next model")
            p.cool(1)
        else:
            raise e