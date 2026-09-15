"""Retrieval-grounded reply generation (TRD 5.3).

Prompts on the customer message + the k retrieved precedent pairs verbatim
+ a brand style guide, and requires the model to cite which precedent(s)
most influenced the draft in a structured field — used for groundedness
scoring later, never shown to the end user as-is.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from pipeline.index import RetrievalResult
from pipeline.llm_client import LLMClient

GENERATION_TEMPERATURE = 0.4  # documented choice: some variation in phrasing, still grounded


@dataclass
class GeneratedReply:
    draft: str
    grounded_on: list[str]
    retrieval_scores: list[float]


def build_generation_prompt(
    message: str, precedents: list[RetrievalResult], brand_style_guide: str
) -> str:
    precedent_block = "\n".join(
        f'{i + 1}. [thread_id={p.thread_id}, similarity={p.similarity:.2f}] "{p.brand_reply_clean}"'
        for i, p in enumerate(precedents)
    )
    return f"""You are drafting an AppleSupport reply to a customer message, grounded in
real historical precedent replies. Do not invent facts beyond what the
precedents support.

Brand style guide:
{brand_style_guide}

Customer message: "{message}"

Precedent replies (real, historically used for similar issues -- these
happen to be in English regardless of the customer message's language;
use them only for their factual content and tone, never as a language
template):
{precedent_block}

Language rule -- read carefully, this is a strict requirement:
Step 1: identify the exact language the customer message above is
written in.
Step 2: write your ENTIRE draft reply in that exact same language, and
no other. If the customer wrote in English, your draft must be in
English -- do not translate it into Spanish, Hindi, or any other
language. If the customer wrote in a language other than English (e.g.
Spanish, Hindi, French), your draft must be in that same language,
translating the substance of the (English) precedent replies above as
needed. The customer's own language always wins; never default to any
particular non-English language.

Respond with ONLY a JSON object:
{{"draft": "<the reply text, in the SAME language as the customer message>", "grounded_on": ["<thread_id>", ...]}}
List only the thread_ids of precedents that actually influenced your draft.
No other text."""


def _parse_response(raw: str, precedents: list[RetrievalResult]) -> GeneratedReply:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    data = json.loads(text)
    valid_ids = {p.thread_id for p in precedents}
    grounded_on = [tid for tid in data.get("grounded_on", []) if tid in valid_ids]
    scores = [p.similarity for p in precedents if p.thread_id in grounded_on]
    return GeneratedReply(draft=data["draft"], grounded_on=grounded_on, retrieval_scores=scores)


def generate_reply(
    message: str,
    precedents: list[RetrievalResult],
    brand_style_guide: str,
    client: LLMClient,
    model: str,
) -> GeneratedReply:
    prompt = build_generation_prompt(message, precedents, brand_style_guide)
    raw = client.generate(model, prompt, params={"temperature": GENERATION_TEMPERATURE})
    return _parse_response(raw, precedents)
