import numpy as np
import pandas as pd

from pipeline.index import build_index


def _fake_embed_fn(messages):
    # deterministic, content-based fake embedding: hash each message into a small vector
    rng_seed = lambda s: abs(hash(s)) % (2**32)
    vectors = []
    for msg in messages:
        rng = np.random.default_rng(rng_seed(msg))
        vectors.append(rng.normal(size=4))
    return np.asarray(vectors)


def test_build_index_excludes_unresolved_threads():
    threads = pd.DataFrame(
        {
            "thread_id": ["t1", "t2", "t3"],
            "customer_msg_clean": ["help me please", "billing issue", "unresolved case"],
            "brand_reply_clean": ["reply1", "reply2", "reply3"],
            "intent": ["a", "b", "a"],
            "resolved": [True, True, False],
        }
    )
    index = build_index(threads, _fake_embed_fn)
    assert len(index) == 2
    assert "t3" not in index.metadata["thread_id"].tolist()


def test_build_index_roundtrips_save_load(tmp_path):
    threads = pd.DataFrame(
        {
            "thread_id": ["t1", "t2"],
            "customer_msg_clean": ["help me please", "billing issue"],
            "brand_reply_clean": ["reply1", "reply2"],
            "intent": ["a", "b"],
            "resolved": [True, True],
        }
    )
    index = build_index(threads, _fake_embed_fn)
    emb_path = tmp_path / "emb.npy"
    meta_path = tmp_path / "meta.parquet"
    index.save(emb_path, meta_path)

    from pipeline.index import RetrievalIndex

    reloaded = RetrievalIndex.load(emb_path, meta_path)
    assert len(reloaded) == len(index)
    assert reloaded.metadata["thread_id"].tolist() == index.metadata["thread_id"].tolist()
