"""Main intent classifier: few-shot prompted LLM using taxonomy.yaml (TRD 4.2, Option A).

taxonomy.yaml is the single source of truth for labels, descriptions, and
example utterances (TRD 3.5) — this module never invents or hardcodes intent
names, it reads them from that file so relabeling the taxonomy doesn't
require touching classifier code.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from pipeline.config import TAXONOMY_YAML
from pipeline.llm_client import LLMClient

CLASSIFIER_TEMPERATURE = 0.0  # documented low-temperature choice, PRD 5


@dataclass
class ClassificationResult:
    intent: str
    confidence: float
    all_scores: dict[str, float]


def load_taxonomy(path: Path = TAXONOMY_YAML) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["intents"]


def build_classification_prompt(message: str, taxonomy: list[dict]) -> str:
    intent_block = "\n".join(
        f"- {intent['id']}: {intent['description']} "
        f"(examples: {'; '.join(intent.get('examples', [])[:3])})"
        for intent in taxonomy
    )
    valid_ids = [intent["id"] for intent in taxonomy]
    return f"""You are classifying a customer support message into exactly one intent.

Intents:
{intent_block}

Message: "{message}"

Respond with ONLY a JSON object of the form:
{{"intent": "<one of {valid_ids}>", "confidence": <float 0-1>, "all_scores": {{"<intent_id>": <float>, ...}}}}
No other text."""


def _parse_response(raw: str, valid_ids: list[str]) -> ClassificationResult:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    data = json.loads(text)
    intent = data["intent"]
    if intent not in valid_ids:
        raise ValueError(f"model returned invalid intent {intent!r}, expected one of {valid_ids}")
    return ClassificationResult(
        intent=intent,
        confidence=float(data["confidence"]),
        all_scores={k: float(v) for k, v in data.get("all_scores", {}).items()},
    )


def classify(
    message: str,
    client: LLMClient,
    model: str,
    taxonomy: list[dict] | None = None,
) -> ClassificationResult:
    taxonomy = taxonomy if taxonomy is not None else load_taxonomy()
    valid_ids = [intent["id"] for intent in taxonomy]
    prompt = build_classification_prompt(message, taxonomy)
    raw = client.generate(model, prompt, params={"temperature": CLASSIFIER_TEMPERATURE})
    return _parse_response(raw, valid_ids)
