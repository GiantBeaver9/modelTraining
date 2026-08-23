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


def _looks_like_key(s: str) -> bool:
    """Heuristic: does this string look like an API KEY rather than an env-var NAME?"""
    return bool(s) and (len(s) > 30 or "." in s or s != s.upper())


def _resolve_key(model_cfg: dict) -> str:
    """Resolve the API key for a model.

    Two ways, in priority order:
      1. Direct value: model_cfg["api_key"]  (e.g. app.py's GUARD_API_KEY / JUDGE_API_KEY).
      2. Indirect: model_cfg["api_key_env"] names the ENV VAR that holds the key.
    """
    direct = model_cfg.get("api_key")
    if direct:
        return direct
    env_var = model_cfg.get("api_key_env")
    if not env_var:
        raise ConfigError(f"Model {model_cfg.get('id')!r} has neither 'api_key' nor 'api_key_env'.")
    key = os.environ.get(env_var)
    if key:
        return key
    hint = ""
    if _looks_like_key(env_var):
        hint = ("  It looks like you set the key VALUE as the variable NAME. api_key_env must be the "
                "NAME of an env var (e.g. GEMINI_API_KEY) — put the key in that variable, or pass the "
                "key directly (GUARD_API_KEY / JUDGE_API_KEY / cfg 'api_key').")
    raise ConfigError(f"Environment variable {env_var!r} is not set.{hint}")


def _retry(fn, *, attempts: int = 7, base_delay: float = 2.0, max_delay: float = 30.0):
    """Call fn() with exponential backoff on transient errors.

    Patient enough to ride out a dedicated HF Inference Endpoint's COLD START: an endpoint with
    scale-to-zero returns 503 / "initializing" / "loading" for up to a few minutes on the first
    request after idle. 7 attempts with capped backoff spans ~2+4+8+16+30+30 ≈ 90s of warm-up
    before giving up, so the demo self-heals instead of surfacing "model call failed".
    """
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
                          "502", "500", "429", "connection", "unavailable",
                          # cold-start / endpoint-warming signals
                          "initializ", "loading", "starting", "not ready", "scal",
                          "temporarily", "bad gateway")
            )
            if not transient or i == attempts - 1:
                raise
            time.sleep(min(base_delay * (2 ** i), max_delay))
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
        self._client = genai.Client(api_key=_resolve_key(model_cfg))

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
        kwargs: dict[str, Any] = {"api_key": _resolve_key(model_cfg)}
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
        self._client = Anthropic(api_key=_resolve_key(model_cfg))

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


class _HFClient:
    """Local inference of a Hugging Face repo id via transformers (the graded `--model <hf-repo-id>`
    path). Needs `transformers` + `torch` installed and, realistically, a GPU. No API key."""
    _cache: dict = {}

    def __init__(self, model_cfg: dict):
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        self.model_id = model_cfg["model"]
        if self.model_id not in _HFClient._cache:
            tok = AutoTokenizer.from_pretrained(self.model_id)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id, torch_dtype="auto", device_map="auto"
            )
            _HFClient._cache[self.model_id] = (tok, model)
        self.tok, self.model = _HFClient._cache[self.model_id]

    def chat(self, messages: list[dict], temperature: float, max_tokens: int) -> str:
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs, max_new_tokens=max_tokens,
            do_sample=temperature > 0, temperature=max(temperature, 1e-5),
            pad_token_id=self.tok.eos_token_id,
        )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tok.decode(gen, skip_special_tokens=True).strip()


_PROVIDERS = {
    "gemini": _GeminiClient,
    "openai_compatible": _OpenAICompatibleClient,
    "anthropic": _AnthropicClient,
    "hf": _HFClient,
}


def get_client(model_cfg: dict):
    provider = model_cfg.get("provider")
    if provider not in _PROVIDERS:
        raise ConfigError(
            f"Unknown provider {provider!r} for model {model_cfg.get('id')!r}. "
            f"Choose one of: {', '.join(_PROVIDERS)}"
        )
    return _PROVIDERS[provider](model_cfg)
