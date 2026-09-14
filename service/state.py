"""Loads frozen offline-pipeline artifacts once at service startup.

Stateless per request otherwise (Architecture 2.2) — thread context is
passed in by the caller, nothing about a request is retained server-side.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

from pipeline.classify import load_taxonomy
from pipeline.index import RetrievalIndex
from pipeline.llm_client import LLMClient

DEFAULT_BRAND_STYLE_GUIDE = (
    "Warm but concise. Acknowledge the issue in one short sentence, give a "
    "concrete next step or fix, and offer to follow up via DM if more "
    "account-specific detail is needed. No corporate filler, no over-apologizing."
)


@dataclass
class AppState:
    taxonomy: list[dict]
    retrieval_index: RetrievalIndex
    embed_fn: Callable[[list[str]], "object"]
    llm_client: LLMClient
    brand_style_guide: str
    classifier_model: str
    generation_model: str


def load_state() -> AppState:
    from pipeline.taxonomy import embed_messages

    taxonomy = load_taxonomy()
    retrieval_index = RetrievalIndex.load()
    llm_client = LLMClient()

    return AppState(
        taxonomy=taxonomy,
        retrieval_index=retrieval_index,
        embed_fn=embed_messages,
        llm_client=llm_client,
        brand_style_guide=DEFAULT_BRAND_STYLE_GUIDE,
        classifier_model=os.environ.get("CLASSIFIER_MODEL", "gemini-3.5-flash-lite"),
        generation_model=os.environ.get("GENERATION_MODEL", "gemini-3.5-flash-lite"),
    )
