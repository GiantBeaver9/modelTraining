"""Thin, uniform chat clients for the ablation.

Every provider exposes the same interface:

    client = get_client(model_cfg)
    reply  = client.chat(messages, temperature=..., max_tokens=...)

`messages` is a list of {"role": "system"|"user"|"assistant", "content": str}.
`chat` returns the assistant's reply text (str).

Providers (set per model in config.yaml under `provider`):
  - "gemini"            -> native google-genai SDK   (spends your Google/Gemini credit)
  - "openai_compatible" -> openai SDK with base_url   (OpenAI direct, OpenRouter, LiteLLM, ...)
  - "anthropic"         -> native anthropic SDK

SDKs are imported lazily, so you only need the ones you actually use installed.
"""

from __future__ import annotations

import os
import time
from typing import Any


class ConfigError(RuntimeError):
    pass


def _require_key(env_var: str) -> str:
    key = os.environ.get(env_var)
    if not key:
        raise ConfigError(
            f"Environment variable {env_var!r} is not set. Export it before running "
            f"(see config.yaml -> models[].api_key_env)."
        )
    return key


def _retry(fn, *, attempts: int = 4, base_delay: float = 2.0):
    """Call fn() with exponential backoff on transient errors."""
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - providers raise many distinct types
            last = exc
            msg = str(exc).lower()
            transient = any(
                s in msg
                for s in ("rate", "timeout", "timed out", "overload", "503",
                          "502", "500", "429", "connection", "unavailable")
            )
            if not transient or i == attempts - 1:
                raise
            time.sleep(base_delay * (2 ** i))
    raise last  # pragma: no cover


def _split_system(messages: list[dict]) -> tuple[str, list[dict]]:
    """Return (system_text, non_system_messages)."""
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    rest = [m for m in messages if m["role"] != "system"]
    return "\n\n".join(system_parts), rest


class _GeminiClient:
    def __init__(self, model_cfg: dict):
        from google import genai  # type: ignore

        self._genai = genai
        self.model = model_cfg["model"]
        self._client = genai.Client(api_key=_require_key(model_cfg["api_key_env"]))

    def chat(self, messages: list[dict], temperature: float, max_tokens: int) -> str:
        from google.genai import types  # type: ignore

        system_text, rest = _split_system(messages)
        contents = []
        for m in rest:
            role = "model" if m["role"] == "assistant" else "user"
            contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))

        cfg = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system_text or None,
        )

        def _call():
            return self._client.models.generate_content(
                model=self.model, contents=contents, config=cfg
            )

        resp = _retry(_call)
        return (getattr(resp, "text", None) or "").strip()


class _OpenAICompatibleClient:
    def __init__(self, model_cfg: dict):
        from openai import OpenAI  # type: ignore

        self.model = model_cfg["model"]
        kwargs: dict[str, Any] = {"api_key": _require_key(model_cfg["api_key_env"])}
        if model_cfg.get("base_url"):
            kwargs["base_url"] = model_cfg["base_url"]
        self._client = OpenAI(**kwargs)

    def chat(self, messages: list[dict], temperature: float, max_tokens: int) -> str:
        def _call():
            return self._client.chat.completions.create(
                model=self.model,
                messages=messages,  # OpenAI chat format already matches our schema
                temperature=temperature,
                max_tokens=max_tokens,
            )

        resp = _retry(_call)
        return (resp.choices[0].message.content or "").strip()


class _AnthropicClient:
    def __init__(self, model_cfg: dict):
        from anthropic import Anthropic  # type: ignore

        self.model = model_cfg["model"]
        self._client = Anthropic(api_key=_require_key(model_cfg["api_key_env"]))

    def chat(self, messages: list[dict], temperature: float, max_tokens: int) -> str:
        system_text, rest = _split_system(messages)
        anth_messages = [{"role": m["role"], "content": m["content"]} for m in rest]

        def _call():
            return self._client.messages.create(
                model=self.model,
                system=system_text or None,
                messages=anth_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        resp = _retry(_call)
        parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        return "".join(parts).strip()


_PROVIDERS = {
    "gemini": _GeminiClient,
    "openai_compatible": _OpenAICompatibleClient,
    "anthropic": _AnthropicClient,
}


def get_client(model_cfg: dict):
    provider = model_cfg.get("provider")
    if provider not in _PROVIDERS:
        raise ConfigError(
            f"Unknown provider {provider!r} for model {model_cfg.get('id')!r}. "
            f"Choose one of: {', '.join(_PROVIDERS)}"
        )
    return _PROVIDERS[provider](model_cfg)
