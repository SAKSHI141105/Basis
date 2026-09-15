import json

from pipeline.generate import _parse_response, build_generation_prompt, generate_reply
from pipeline.index import RetrievalResult
from pipeline.llm_client import DiskCache, LLMClient, _cache_key

PRECEDENTS = [
    RetrievalResult(thread_id="t1", brand_reply_clean="Try restarting your device.", intent="device_help", similarity=0.82),
    RetrievalResult(thread_id="t2", brand_reply_clean="Please DM your serial number.", intent="device_help", similarity=0.61),
]
STYLE_GUIDE = "Be concise, empathetic, and sign off with a helpful next step."


def test_build_prompt_includes_precedents_and_message():
    prompt = build_generation_prompt("my phone won't turn on", PRECEDENTS, STYLE_GUIDE)
    assert "my phone won't turn on" in prompt
    assert "Try restarting your device." in prompt
    assert "t1" in prompt and "t2" in prompt


def test_build_prompt_instructs_replying_in_customer_language():
    prompt = build_generation_prompt("mi iPhone no enciende", PRECEDENTS, STYLE_GUIDE)
    assert "mi iPhone no enciende" in prompt
    assert "SAME language as the customer message" in prompt


def test_build_prompt_explicitly_covers_english_case_not_just_non_english():
    # Regression guard: an earlier version of this prompt only talked about
    # translating INTO another language ("do not reply in English if the
    # customer did not write in English") with no symmetric instruction for
    # the English case -- that asymmetric phrasing caused real English
    # customer messages to come back drafted in Spanish (observed live).
    # The prompt must explicitly say to stay in English when the customer
    # wrote in English, not just describe the non-English branch.
    prompt = build_generation_prompt("my phone won't turn on", PRECEDENTS, STYLE_GUIDE)
    normalized = " ".join(prompt.split())
    assert "wrote in English" in normalized
    assert "must be in English" in normalized


def test_parse_response_filters_grounded_on_to_valid_ids():
    raw = json.dumps({"draft": "Please restart your device and let us know.", "grounded_on": ["t1", "bogus_id"]})
    result = _parse_response(raw, PRECEDENTS)
    assert result.grounded_on == ["t1"]
    assert result.retrieval_scores == [0.82]


def test_parse_response_strips_markdown_fence():
    raw = "```json\n" + json.dumps({"draft": "hello", "grounded_on": ["t2"]}) + "\n```"
    result = _parse_response(raw, PRECEDENTS)
    assert result.draft == "hello"
    assert result.grounded_on == ["t2"]


def test_generate_reply_uses_cached_llm_response(tmp_path):
    cache = DiskCache(tmp_path / "cache.jsonl")
    client = LLMClient(cache=cache, mode="fast")
    message = "my phone won't turn on"
    prompt = build_generation_prompt(message, PRECEDENTS, STYLE_GUIDE)
    key = _cache_key("gemini-2.5-flash", prompt, {"temperature": 0.4})
    cached = json.dumps({"draft": "Try a force restart, then contact us if it persists.", "grounded_on": ["t1"]})
    cache.set(key, "gemini-2.5-flash", prompt, {"temperature": 0.4}, cached)

    result = generate_reply(message, PRECEDENTS, STYLE_GUIDE, client, "gemini-2.5-flash")
    assert "force restart" in result.draft
    assert result.grounded_on == ["t1"]
