"""Thin, optional LLM layer (Claude via the Messages API).

Design rules: the platform must work with NO API key and NO network. Every caller has a deterministic local fallback,
every LLM answer is validated before use, and answers are cached on disk so a repeated request is free and reproducible.
Only column names and a small capped sample are ever sent; callers decide what goes into a prompt.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Protocol

API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
PROMPT_VERSION = "1"


class LLMError(RuntimeError):
    """The model was unreachable or returned something unusable. Callers fall back to local generators."""


class LLMClient(Protocol):
    name: str

    def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> str: ...


class AnthropicClient:
    def __init__(self, api_key: str, model: str | None = None, timeout: float = 30.0, url: str = API_URL) -> None:
        self.api_key = api_key
        self.name = model or os.environ.get("SDP_LLM_MODEL", DEFAULT_MODEL)
        self.timeout = timeout
        self.url = url

    def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> str:
        body = json.dumps({"model": self.name, "max_tokens": max_tokens, "temperature": 0, "system": system,
                           "messages": [{"role": "user", "content": prompt}]}).encode("utf-8")
        req = urllib.request.Request(self.url, data=body, method="POST", headers={
            "content-type": "application/json", "x-api-key": self.api_key, "anthropic-version": "2023-06-01"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:  # noqa: S310 - fixed https endpoint
                payload = json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
            raise LLMError(f"LLM request failed: {type(e).__name__}") from e  # never echo headers/keys
        parts = [b.get("text", "") for b in payload.get("content", []) if b.get("type") == "text"]
        if not parts:
            raise LLMError("LLM returned no text")
        return "".join(parts)


_override: LLMClient | None | bool = False  # False = not overridden; None = force "unavailable"


def set_client(client: LLMClient | None | bool) -> None:
    """Tests and embedders inject a client (or None to force the offline path). `False` restores the default."""
    global _override
    _override = client


def get_client() -> LLMClient | None:
    if _override is not False:
        return _override  # type: ignore[return-value]
    key = os.environ.get("ANTHROPIC_API_KEY")
    return AnthropicClient(key) if key else None


_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> Any:
    """Parse the first JSON value in a model reply, tolerating code fences and surrounding prose."""
    m = _JSON_BLOCK.search(text)
    candidates = [m.group(1)] if m else []
    candidates.append(text)
    dec = json.JSONDecoder()
    for c in candidates:
        for i, ch in enumerate(c):
            if ch in "[{":
                try:
                    return dec.raw_decode(c[i:])[0]
                except ValueError:
                    continue
    raise LLMError("no JSON found in the model reply")


def cache_dir() -> Path:
    root = Path(os.environ.get("SDP_DATA_DIR") or Path(__file__).resolve().parents[2] / "data")
    d = root / "llm_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_key(client_name: str, system: str, prompt: str) -> str:
    return hashlib.sha256(json.dumps([PROMPT_VERSION, client_name, system, prompt], ensure_ascii=False).encode()).hexdigest()


def cached_complete(client: LLMClient, system: str, prompt: str, max_tokens: int = 1024, use_cache: bool = True) -> tuple[str, bool]:
    """Returns (text, from_cache). Cache lives under SDP_DATA_DIR/llm_cache; a corrupt entry is ignored."""
    key = cache_key(client.name, system, prompt)
    path = cache_dir() / f"{key}.json"
    if use_cache and path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))["text"], True
        except (ValueError, KeyError, OSError):
            pass
    text = client.complete(system, prompt, max_tokens)
    if use_cache:
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)
        except OSError:
            pass
    return text, False
