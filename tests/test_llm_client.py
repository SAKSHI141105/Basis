import pytest

from pipeline.llm_client import DiskCache, LLMClient, _cache_key


def test_cache_key_deterministic():
    k1 = _cache_key("m", "prompt", {"temperature": 0})
    k2 = _cache_key("m", "prompt", {"temperature": 0})
    assert k1 == k2


def test_cache_key_changes_with_prompt():
    k1 = _cache_key("m", "prompt a", {})
    k2 = _cache_key("m", "prompt b", {})
    assert k1 != k2


def test_disk_cache_roundtrip(tmp_path):
    cache = DiskCache(tmp_path / "cache.jsonl")
    cache.set("key1", "m", "p", {}, "response text")
    assert cache.get("key1") == "response text"

    reloaded = DiskCache(tmp_path / "cache.jsonl")
    assert reloaded.get("key1") == "response text"


def test_fast_mode_cache_hit_returns_cached_response(tmp_path):
    cache = DiskCache(tmp_path / "cache.jsonl")
    client = LLMClient(cache=cache, mode="fast")
    key = _cache_key("gemini-2.5-flash-lite", "hello", {})
    cache.set(key, "gemini-2.5-flash-lite", "hello", {}, "cached reply")

    result = client.generate("gemini-2.5-flash-lite", "hello", {})
    assert result == "cached reply"


def test_fast_mode_cache_miss_raises(tmp_path):
    cache = DiskCache(tmp_path / "cache.jsonl")
    client = LLMClient(cache=cache, mode="fast")

    with pytest.raises(RuntimeError, match="Cache miss in fast mode"):
        client.generate("gemini-2.5-flash-lite", "never cached", {})
