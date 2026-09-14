"""Freeze the committed fast-mode LLM cache from the accumulated live cache.

Run once after a full `make eval-live` run completes: copies every
{key, model, prompt, params, response} record from the growing (gitignored)
live cache into the small, committed cache that `make eval-fast` replays
(TRD 8.2, AGENT_WORKING_AGREEMENT §2 — the one deliberate exception to
"never commit .cache/").
"""
from __future__ import annotations

import shutil

from pipeline.llm_client import DEFAULT_CACHE_PATH, GOLDEN_CACHE_PATH


def run(source=DEFAULT_CACHE_PATH, dest=GOLDEN_CACHE_PATH) -> int:
    if not source.exists():
        raise FileNotFoundError(f"no live cache found at {source} — run `make eval-live` first")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source, dest)
    with open(dest, "r", encoding="utf-8") as f:
        n = sum(1 for _ in f)
    return n


if __name__ == "__main__":
    n = run()
    print(f"wrote {n} cached responses to {GOLDEN_CACHE_PATH}")
