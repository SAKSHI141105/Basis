import numpy as np
import pandas as pd
import pytest

from pipeline.index import RetrievalIndex


def _toy_index():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ],
        dtype=np.float32,
    )
    metadata = pd.DataFrame(
        {
            "thread_id": ["t1", "t2", "t3", "t4"],
            "brand_reply_clean": ["reply1", "reply2", "reply3", "reply4"],
            "intent": ["billing", "billing", "device_help", "device_help"],
        }
    )
    return RetrievalIndex(embeddings, metadata)


def test_query_returns_expected_nearest_neighbor():
    index = _toy_index()
    results = index.query(np.array([1.0, 0.05]), k=1)
    assert results[0].thread_id in {"t1", "t2"}


def test_query_respects_intent_filter():
    index = _toy_index()
    results = index.query(np.array([0.0, 1.0]), k=2, intent="billing")
    assert all(r.intent == "billing" for r in results)


def test_query_falls_back_when_intent_has_no_candidates():
    index = _toy_index()
    results = index.query(np.array([1.0, 0.0]), k=1, intent="nonexistent_intent")
    assert len(results) == 1


def test_mismatched_lengths_raises():
    embeddings = np.zeros((3, 2))
    metadata = pd.DataFrame({"thread_id": ["a", "b"], "brand_reply_clean": ["x", "y"]})
    with pytest.raises(ValueError):
        RetrievalIndex(embeddings, metadata)
