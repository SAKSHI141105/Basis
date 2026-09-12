import json

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from pipeline.classify import build_classification_prompt
from pipeline.generate import build_generation_prompt
from pipeline.index import RetrievalIndex
from pipeline.llm_client import DiskCache, LLMClient, _cache_key
from service.main import create_app
from service.state import AppState

TAXONOMY = [
    {"id": "device_help", "description": "device troubleshooting", "examples": ["phone won't turn on"]},
    {"id": "billing", "description": "billing issues", "examples": ["overcharged"]},
]


def _fake_embed_fn(messages):
    vectors = []
    for msg in messages:
        rng = np.random.default_rng(abs(hash(msg)) % (2**32))
        vectors.append(rng.normal(size=4))
    return np.asarray(vectors)


def _build_fake_state(tmp_path) -> AppState:
    embeddings = _fake_embed_fn(["my phone froze", "billing question"])
    metadata = pd.DataFrame(
        {
            "thread_id": ["t1", "t2"],
            "brand_reply_clean": ["Try restarting your device.", "We'll review your billing statement."],
            "intent": ["device_help", "billing"],
        }
    )
    index = RetrievalIndex(embeddings, metadata)

    cache = DiskCache(tmp_path / "cache.jsonl")
    client = LLMClient(cache=cache, mode="fast")

    message = "my iphone screen is frozen and won't respond"
    classify_prompt = build_classification_prompt(message, TAXONOMY)
    classify_key = _cache_key("gemini-2.5-flash-lite", classify_prompt, {"temperature": 0.0})
    cache.set(
        classify_key,
        "gemini-2.5-flash-lite",
        classify_prompt,
        {"temperature": 0.0},
        json.dumps({"intent": "device_help", "confidence": 0.9, "all_scores": {"device_help": 0.9, "billing": 0.05}}),
    )

    precedents = index.query(np.asarray(_fake_embed_fn([message])[0]), k=2, intent="device_help")
    gen_prompt = build_generation_prompt(
        message, precedents, "Warm but concise, one concrete next step."
    )
    gen_key = _cache_key("gemini-2.5-flash", gen_prompt, {"temperature": 0.4})
    cache.set(
        gen_key,
        "gemini-2.5-flash",
        gen_prompt,
        {"temperature": 0.4},
        json.dumps({"draft": "Try a force restart; let us know if that doesn't work.", "grounded_on": [precedents[0].thread_id]}),
    )

    return AppState(
        taxonomy=TAXONOMY,
        retrieval_index=index,
        embed_fn=_fake_embed_fn,
        llm_client=client,
        brand_style_guide="Warm but concise, one concrete next step.",
        classifier_model="gemini-2.5-flash-lite",
        generation_model="gemini-2.5-flash",
    ), message


def test_health_endpoint(tmp_path):
    state, _ = _build_fake_state(tmp_path)
    app = create_app(state=state)
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_classify_endpoint_shape(tmp_path):
    state, message = _build_fake_state(tmp_path)
    app = create_app(state=state)
    with TestClient(app) as client:
        resp = client.post("/classify", json={"message": message})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"intent", "confidence", "all_scores"}
    assert isinstance(body["intent"], str)
    assert isinstance(body["confidence"], float)
    assert isinstance(body["all_scores"], dict)


def test_draft_reply_endpoint_shape(tmp_path):
    state, message = _build_fake_state(tmp_path)
    app = create_app(state=state)
    with TestClient(app) as client:
        resp = client.post("/draft-reply", json={"message": message, "intent": "device_help"})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"draft", "grounded_on", "retrieval_scores"}
    assert isinstance(body["draft"], str)
    assert isinstance(body["grounded_on"], list)
    assert isinstance(body["retrieval_scores"], list)


def test_decide_endpoint_shape(tmp_path):
    state, message = _build_fake_state(tmp_path)
    app = create_app(state=state)
    with TestClient(app) as client:
        resp = client.post("/decide", json={"message": message})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"decision", "reason", "signals"}
    assert body["decision"] in {"auto_handle", "escalate"}
    assert isinstance(body["reason"], str)


def test_pipeline_endpoint_shape(tmp_path):
    state, message = _build_fake_state(tmp_path)
    app = create_app(state=state)
    with TestClient(app) as client:
        resp = client.post("/pipeline", json={"message": message})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"classify", "draft_reply", "decision"}
    assert body["classify"]["intent"] == "device_help"
    assert body["decision"]["decision"] in {"auto_handle", "escalate"}
