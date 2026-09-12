import json

import pytest

from pipeline.classify import _parse_response, build_classification_prompt, classify
from pipeline.llm_client import DiskCache, LLMClient, _cache_key

TAXONOMY = [
    {"id": "billing", "description": "billing and subscription issues", "examples": ["I was overcharged"]},
    {"id": "device_help", "description": "device troubleshooting", "examples": ["phone won't turn on"]},
    {"id": "out_of_scope", "description": "unrelated or spam", "examples": ["random unrelated text"]},
]


def test_build_prompt_includes_all_intents_and_message():
    prompt = build_classification_prompt("my phone is broken", TAXONOMY)
    assert "billing" in prompt
    assert "device_help" in prompt
    assert "out_of_scope" in prompt
    assert "my phone is broken" in prompt


def test_parse_response_valid_json():
    raw = json.dumps({"intent": "billing", "confidence": 0.87, "all_scores": {"billing": 0.87, "device_help": 0.1}})
    result = _parse_response(raw, ["billing", "device_help", "out_of_scope"])
    assert result.intent == "billing"
    assert result.confidence == 0.87
    assert result.all_scores["device_help"] == 0.1


def test_parse_response_strips_markdown_fence():
    raw = "```json\n" + json.dumps({"intent": "device_help", "confidence": 0.5, "all_scores": {}}) + "\n```"
    result = _parse_response(raw, ["billing", "device_help"])
    assert result.intent == "device_help"


def test_parse_response_rejects_invalid_intent():
    raw = json.dumps({"intent": "not_a_real_intent", "confidence": 0.9, "all_scores": {}})
    with pytest.raises(ValueError):
        _parse_response(raw, ["billing", "device_help"])


def test_classify_uses_cached_llm_response(tmp_path):
    cache = DiskCache(tmp_path / "cache.jsonl")
    client = LLMClient(cache=cache, mode="fast")
    message = "my phone screen is cracked"
    prompt = build_classification_prompt(message, TAXONOMY)
    key = _cache_key("gemini-2.5-flash-lite", prompt, {"temperature": 0.0})
    cached_response = json.dumps({"intent": "device_help", "confidence": 0.92, "all_scores": {"device_help": 0.92}})
    cache.set(key, "gemini-2.5-flash-lite", prompt, {"temperature": 0.0}, cached_response)

    result = classify(message, client, "gemini-2.5-flash-lite", taxonomy=TAXONOMY)
    assert result.intent == "device_help"
    assert result.confidence == 0.92
