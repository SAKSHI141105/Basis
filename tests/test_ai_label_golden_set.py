import json

import pandas as pd

from pipeline.ai_label_golden_set import build_labeling_prompt, _parse_response, label_example
from pipeline.llm_client import DiskCache, LLMClient, _cache_key

INTENTS = [
    {"id": "device_help", "description": "device troubleshooting"},
    {"id": "billing", "description": "billing issues"},
]


def test_build_prompt_includes_message_and_intents():
    prompt = build_labeling_prompt("my phone is broken", "reply text", None, 2, INTENTS)
    assert "my phone is broken" in prompt
    assert "device_help" in prompt
    assert "billing" in prompt


def test_build_prompt_includes_followup_when_present():
    prompt = build_labeling_prompt("msg", "reply", "still broken", 3, INTENTS)
    assert "still broken" in prompt


def test_parse_response_valid():
    raw = json.dumps({"true_intent": "billing", "true_escalation": "escalate", "ambiguous_note": ""})
    result = _parse_response(raw, ["billing", "device_help"])
    assert result["true_intent"] == "billing"
    assert result["true_escalation"] == "escalate"
    assert result["label_source"] == "ai_generated"


def test_label_example_uses_cached_response(tmp_path):
    cache = DiskCache(tmp_path / "cache.jsonl")
    client = LLMClient(cache=cache, mode="fast")
    row = pd.Series({
        "customer_msg_clean": "my phone froze",
        "brand_reply_clean": "try restarting",
        "followup_msg": None,
        "turn_count": 2,
    })
    prompt = build_labeling_prompt(row["customer_msg_clean"], row["brand_reply_clean"], None, 2, INTENTS)
    key = _cache_key("gemini-3.5-flash", prompt, {"temperature": 0.0})
    cache.set(key, "gemini-3.5-flash", prompt, {"temperature": 0.0},
              json.dumps({"true_intent": "device_help", "true_escalation": "auto_handle", "ambiguous_note": ""}))

    import pipeline.ai_label_golden_set as mod
    mod.LABELING_MODEL = "gemini-3.5-flash"
    result = label_example(row, client, INTENTS)
    assert result["true_intent"] == "device_help"
