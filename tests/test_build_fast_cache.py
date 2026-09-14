import pytest

from pipeline.build_fast_cache import run


def test_run_copies_live_cache_to_golden_cache(tmp_path):
    source = tmp_path / "live.jsonl"
    dest = tmp_path / "golden.jsonl"
    source.write_text('{"key": "a"}\n{"key": "b"}\n', encoding="utf-8")

    n = run(source=source, dest=dest)

    assert n == 2
    assert dest.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")


def test_run_raises_when_no_live_cache_exists(tmp_path):
    with pytest.raises(FileNotFoundError):
        run(source=tmp_path / "nonexistent.jsonl", dest=tmp_path / "golden.jsonl")
