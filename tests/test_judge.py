import json

from eval.judge import _parse_response, build_judge_prompt, judge_reply
from pipeline.index import RetrievalResult
from pipeline.llm_client import DiskCache, LLMClient, _cache_key

PRECEDENTS = [
    RetrievalResult(thread_id="t1", brand_reply_clean="Try restarting your device.", intent="device_help", similarity=0.8),
]


def test_build_judge_prompt_includes_message_draft_and_precedents():
    prompt = build_judge_prompt("phone frozen", "Try a force restart.", PRECEDENTS)
    assert "phone frozen" in prompt
    assert "Try a force restart." in prompt
    assert "Try restarting your device." in prompt


def test_parse_response_extracts_scores():
    raw = json.dumps(
        {"groundedness": 4, "correctness": 5, "tone": 4, "actionability": 5, "rationale": "Solid, grounded reply."}
    )
    score = _parse_response(raw)
    assert score.groundedness == 4
    assert score.mean_score == 4.5


def test_parse_response_strips_markdown_fence():
    raw = "```json\n" + json.dumps(
        {"groundedness": 3, "correctness": 3, "tone": 3, "actionability": 3, "rationale": "ok"}
    ) + "\n```"
    score = _parse_response(raw)
    assert score.mean_score == 3.0


def test_judge_reply_uses_cached_llm_response(tmp_path):
    cache = DiskCache(tmp_path / "cache.jsonl")
    client = LLMClient(cache=cache, mode="fast")
    message = "phone frozen"
    draft = "Try a force restart."
    prompt = build_judge_prompt(message, draft, PRECEDENTS)
    key = _cache_key("gemini-2.5-flash", prompt, {"temperature": 0.0})
    cached = json.dumps({"groundedness": 5, "correctness": 4, "tone": 4, "actionability": 4, "rationale": "Good."})
    cache.set(key, "gemini-2.5-flash", prompt, {"temperature": 0.0}, cached)

    score = judge_reply(message, draft, PRECEDENTS, client, "gemini-2.5-flash")
    assert score.groundedness == 5
