from pipeline.llm_client import LLMClient
import pipeline.llm_client as llm_client_module


def test_fast_mode_defaults_to_golden_cache_path(tmp_path, monkeypatch):
    fast_path = tmp_path / "golden.jsonl"
    monkeypatch.setattr(llm_client_module, "GOLDEN_CACHE_PATH", fast_path)
    client = LLMClient(mode="fast")
    assert client.cache.path == fast_path


def test_live_mode_defaults_to_default_cache_path(tmp_path, monkeypatch):
    live_path = tmp_path / "live.jsonl"
    monkeypatch.setattr(llm_client_module, "DEFAULT_CACHE_PATH", live_path)
    client = LLMClient(mode="live")
    assert client.cache.path == live_path


def test_explicit_cache_overrides_mode_default(tmp_path):
    from pipeline.llm_client import DiskCache

    explicit = DiskCache(tmp_path / "explicit.jsonl")
    client = LLMClient(cache=explicit, mode="fast")
    assert client.cache is explicit
