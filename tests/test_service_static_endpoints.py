import json

from fastapi.testclient import TestClient

import service.main as main_module
from service.main import create_app
from service.state import AppState


def _minimal_state():
    from pipeline.index import RetrievalIndex
    import numpy as np
    import pandas as pd

    index = RetrievalIndex(np.zeros((1, 4)), pd.DataFrame({"thread_id": ["a"], "brand_reply_clean": ["hi"]}))
    return AppState(
        taxonomy=[],
        retrieval_index=index,
        embed_fn=lambda msgs: np.zeros((len(msgs), 4)),
        llm_client=None,
        brand_style_guide="",
        classifier_model="x",
        generation_model="x",
    )


def test_eval_report_endpoint_returns_404_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(main_module, "EVAL_REPORT_PATH", tmp_path / "nonexistent.json")
    app = create_app(state=_minimal_state())
    with TestClient(app) as client:
        resp = client.get("/eval-report")
    assert resp.status_code == 404


def test_eval_report_endpoint_returns_content_when_present(tmp_path, monkeypatch):
    report_path = tmp_path / "eval_report.json"
    report_path.write_text(json.dumps({"golden_set_size": 5}), encoding="utf-8")
    monkeypatch.setattr(main_module, "EVAL_REPORT_PATH", report_path)
    app = create_app(state=_minimal_state())
    with TestClient(app) as client:
        resp = client.get("/eval-report")
    assert resp.status_code == 200
    assert resp.json()["golden_set_size"] == 5


def test_samples_endpoint_returns_real_messages(tmp_path, monkeypatch):
    golden_path = tmp_path / "golden_set.jsonl"
    with open(golden_path, "w", encoding="utf-8") as f:
        for i in range(3):
            f.write(json.dumps({"thread_id": f"t{i}", "customer_msg": f"message {i}"}) + "\n")
    monkeypatch.setattr(main_module, "GOLDEN_SET_PATH", golden_path)
    app = create_app(state=_minimal_state())
    with TestClient(app) as client:
        resp = client.get("/samples")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 3
    assert body[0] == {"thread_id": "t0", "message": "message 0"}


def test_samples_endpoint_caps_at_sample_count(tmp_path, monkeypatch):
    golden_path = tmp_path / "golden_set.jsonl"
    with open(golden_path, "w", encoding="utf-8") as f:
        for i in range(20):
            f.write(json.dumps({"thread_id": f"t{i}", "customer_msg": f"message {i}"}) + "\n")
    monkeypatch.setattr(main_module, "GOLDEN_SET_PATH", golden_path)
    monkeypatch.setattr(main_module, "SAMPLE_COUNT", 5)
    app = create_app(state=_minimal_state())
    with TestClient(app) as client:
        resp = client.get("/samples")
    assert len(resp.json()) == 5


def test_cors_headers_present_for_allowed_origin():
    app = create_app(state=_minimal_state())
    with TestClient(app) as client:
        resp = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:3000"
