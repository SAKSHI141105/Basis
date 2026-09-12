"""Shared, cached, rate-limit-aware Gemini wrapper.

Used by /classify, /draft-reply, /decide, and the eval harness alike (TRD
§8.3) — built early so caching/pacing isn't retrofitted into three
already-written endpoints later.

Every call is wrapped in a disk-backed cache keyed on
hash(model + prompt + params), per TRD §8.2. This makes reruns free/instant
and gives the eval harness a "fast" mode that doesn't depend on live network
calls or free-tier rate limits.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from pipeline.config import CACHE_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CACHE_PATH = CACHE_DIR / "llm_cache.jsonl"

# Approximate free-tier requests-per-minute per model tier (TRD §8.1).
# Exact numbers fluctuate — verify against Google's current rate-limit page
# before relying on these for a live run; these are conservative defaults.
MODEL_RPM = {
    "gemini-2.5-flash-lite": 15,
    "gemini-2.5-flash": 10,
    "gemini-2.5-pro": 5,
}


def _cache_key(model: str, prompt: str, params: dict[str, Any]) -> str:
    payload = json.dumps({"model": model, "prompt": prompt, "params": params}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class DiskCache:
    """Append-only JSONL cache: one line per {key, model, prompt, params, response}."""

    def __init__(self, path: Path = DEFAULT_CACHE_PATH):
        self.path = path
        self._store: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                self._store[record["key"]] = record["response"]

    def get(self, key: str):
        return self._store.get(key)

    def set(self, key: str, model: str, prompt: str, params: dict[str, Any], response: Any) -> None:
        self._store[key] = response
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {"key": key, "model": model, "prompt": prompt, "params": params, "response": response}
                )
                + "\n"
            )

    def __contains__(self, key: str) -> bool:
        return key in self._store


class TokenBucketPacer:
    """Simple sleep-based pacer so a live run degrades gracefully under RPM limits."""

    def __init__(self, rpm: int):
        self.min_interval = 60.0 / max(rpm, 1)
        self._last_call: float | None = None

    def wait(self) -> None:
        if self._last_call is not None:
            elapsed = time.monotonic() - self._last_call
            remaining = self.min_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self._last_call = time.monotonic()


class LLMClient:
    """Cached, rate-limit-aware Gemini client with an offline "fast" mode.

    mode="fast" (default for eval-fast): never calls the network — a cache
    miss raises, because the fast path is only valid against a committed
    cache. mode="live" calls the real API, paced to the model's RPM.
    """

    def __init__(self, cache: DiskCache | None = None, mode: str | None = None):
        self.cache = cache or DiskCache()
        self.mode = mode or os.environ.get("EVAL_MODE", "live")
        self._pacers: dict[str, TokenBucketPacer] = {}

    def _pacer_for(self, model: str) -> TokenBucketPacer:
        if model not in self._pacers:
            self._pacers[model] = TokenBucketPacer(MODEL_RPM.get(model, 10))
        return self._pacers[model]

    def generate(self, model: str, prompt: str, params: dict[str, Any] | None = None) -> Any:
        params = params or {}
        key = _cache_key(model, prompt, params)

        cached = self.cache.get(key)
        if cached is not None:
            return cached

        if self.mode == "fast":
            raise RuntimeError(
                f"Cache miss in fast mode for model={model!r}. "
                "Fast mode only replays the committed golden-set cache — "
                "run `make eval-live` to populate real responses."
            )

        response = self._call_live(model, prompt, params)
        self.cache.set(key, model, prompt, params, response)
        return response

    def _call_live(self, model: str, prompt: str, params: dict[str, Any]) -> str:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return self._call_ollama_fallback(prompt, params)

        from google import genai

        self._pacer_for(model).wait()
        client = genai.Client(api_key=api_key)
        result = client.models.generate_content(model=model, contents=prompt, config=params or None)
        return result.text

    def _call_ollama_fallback(self, prompt: str, params: dict[str, Any]) -> str:
        import requests

        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
        logger.warning("no GEMINI_API_KEY set — falling back to local Ollama model %s", ollama_model)
        resp = requests.post(
            f"{base_url}/api/generate",
            json={"model": ollama_model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["response"]
