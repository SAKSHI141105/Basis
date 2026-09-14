from unittest.mock import MagicMock, patch

import pytest

from pipeline.llm_client import DiskCache, LLMClient


def _make_server_error():
    from google.genai import errors

    resp = MagicMock()
    resp.status_code = 503
    return errors.ServerError(503, {"error": {"message": "UNAVAILABLE"}}, resp)


def test_retries_on_transient_server_error_then_succeeds(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    cache = DiskCache(tmp_path / "cache.jsonl")
    client = LLMClient(cache=cache, mode="live")

    ok_result = MagicMock()
    ok_result.text = "hello"

    call_count = {"n": 0}

    def side_effect(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise _make_server_error()
        return ok_result

    fake_genai_client = MagicMock()
    fake_genai_client.models.generate_content.side_effect = side_effect

    with patch("google.genai.Client", return_value=fake_genai_client), \
         patch("time.sleep"):
        result = client.generate("gemini-3.5-flash-lite", "prompt", {})

    assert result == "hello"
    assert call_count["n"] == 3


def test_raises_after_exhausting_retries(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    cache = DiskCache(tmp_path / "cache.jsonl")
    client = LLMClient(cache=cache, mode="live")

    fake_genai_client = MagicMock()
    fake_genai_client.models.generate_content.side_effect = lambda *a, **k: (_ for _ in ()).throw(
        _make_server_error()
    )

    from google.genai import errors

    with patch("google.genai.Client", return_value=fake_genai_client), \
         patch("time.sleep"):
        with pytest.raises(errors.ServerError):
            client.generate("gemini-3.5-flash-lite", "prompt", {})

    assert fake_genai_client.models.generate_content.call_count == 5
