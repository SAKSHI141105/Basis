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

# Derived from 20 real, resolved AppleSupport replies, not assumed --
# see brand_style_guide.md and pipeline/derive_style_guide.py (TRD 5.3).
DEFAULT_BRAND_STYLE_GUIDE = (
    "Warm but concise (aim for 1-3 short sentences, ~15-40 words). Use "
    '"we," never "I." Acknowledge the issue in one short clause, then '
    "either give a concrete next step/link or invite the customer to DM "
    "for anything requiring account-specific detail -- don't try to solve "
    "account-specific issues in the public reply. No sign-off/initials. "
    "No apology-heavy or corporate filler language."
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
