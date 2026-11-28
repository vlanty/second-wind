"""`LLM` adapter for the Anthropic API (Claude)."""

from __future__ import annotations

import os
from typing import Any

import anthropic

from secondwind.core.ports import LLMRequest, LLMResponse

# Tier -> Claude model.
TIER_MODELS = {
    "small": "claude-haiku-5-5",
    "medium": "claude-sonnet-5-5",
    "large": "claude-opus-5-5",
}

# USD per 1M tokens (input, output), Anthropic API list prices.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}

# Models that accept the server-side refusal fallback (`fallbacks: "default"`).
_FALLBACK_PREFIXES = ("claude-sonnet-5-5", "claude-opus-5", "claude-fable-5")


def supports_fallback(model: str) -> bool:
    return model.startswith(_FALLBACK_PREFIXES)


# Prompt cache, as multiples of the input price: reads per model, writes with the 1-hour TTL.
CACHE_READ = {"claude-haiku-5-5": 0.10, "claude-sonnet-5-5": 0.05, "claude-opus-5-5": 0.05}
CACHE_WRITE_1H = 2.0
CACHE_TTL = "1h"


def cost_usd(model: str, input_tokens: int, output_tokens: int, cache_read: int = 0, cache_write: int = 0) -> float:
    """List-price cost of one call; 0 for a model missing from PRICES (tokens are still counted)."""
    for name, (p_in, p_out) in PRICES.items():
        if model.startswith(name):
            cached = cache_read * CACHE_READ.get(name, 0.1) + cache_write * CACHE_WRITE_1H
            return ((input_tokens + cached) * p_in + output_tokens * p_out) / 1_000_000
    return 0.0


class ClaudeLLM:
    def __init__(self, api_key: str | None = None, client: anthropic.Anthropic | None = None) -> None:
        self._api_key = api_key
        self._client = client

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self._api_key or os.environ["ANTHROPIC_API_KEY"])
        return self._client

    def complete(self, req: LLMRequest) -> LLMResponse:
        model, params = self._params(req)
        if supports_fallback(model):
            response = self.client.beta.messages.create(betas=FALLBACK_BETAS, fallbacks="default", **params)
        else:
            response = self.client.messages.create(**params)
        return _to_response(response)

    def stream(self, req: LLMRequest, on_text) -> LLMResponse:
        model, params = self._params(req)
        ctx: Any
        if supports_fallback(model):
            ctx = self.client.beta.messages.stream(betas=FALLBACK_BETAS, fallbacks="default", **params)
        else:
            ctx = self.client.messages.stream(**params)
        with ctx as s:
            for text in s.text_stream:
                on_text(text)
            return _to_response(s.get_final_message())

    def _params(self, req: LLMRequest) -> tuple[str, dict]:
        model = req.model or TIER_MODELS[req.tier]
        params: dict = dict(model=model, max_tokens=req.max_tokens, messages=_messages(req))
        output_config: dict = {}
        if req.effort:
            output_config["effort"] = req.effort
        if req.schema:
            output_config["format"] = {"type": "json_schema", "schema": req.schema}
        if output_config:
            params["output_config"] = output_config
        if req.timeout:
            params["timeout"] = req.timeout
        if req.system:
            params["system"] = [{"type": "text", "text": b.text,
                                 **({"cache_control": {"type": "ephemeral"}} if b.cache else {})}
                                for b in req.system]
        return model, params


FALLBACK_BETAS = ["server-side-fallback-2026-07-01"]


def _messages(req: LLMRequest) -> list[dict]:
    """The first user message split at cache_prefix, with a cache mark on the stable part."""
    if not req.cache_prefix or not req.messages or not isinstance(req.messages[0].get("content"), str):
        return req.messages
    first, *rest = req.messages
    text = first["content"]
    blocks = [{"type": "text", "text": text[:req.cache_prefix], "cache_control": {"type": "ephemeral", "ttl": CACHE_TTL}}]
    if text[req.cache_prefix:]:
        blocks.append({"type": "text", "text": text[req.cache_prefix:]})
    return [{**first, "content": blocks}, *rest]


def _to_response(response) -> LLMResponse:
    text = "\n".join(b.text for b in response.content if b.type == "text").strip()
    u = response.usage
    read = getattr(u, "cache_read_input_tokens", 0) or 0
    write = getattr(u, "cache_creation_input_tokens", 0) or 0
    return LLMResponse(text=text, model=response.model, input_tokens=u.input_tokens,
                       output_tokens=u.output_tokens, cache_read_tokens=read, cache_write_tokens=write,
                       cost_usd=cost_usd(response.model, u.input_tokens, u.output_tokens, read, write),
                       stop_reason=response.stop_reason, refused=response.stop_reason == "refusal",
                       refusal_details=getattr(response, "stop_details", None))
