import pandas as pd

from pipeline.split import split_threads


def _fixture_threads(n_per_intent=20):
    rows = []
    for intent in ["billing", "device_help", "software"]:
        for i in range(n_per_intent):
            rows.append(
                {
                    "thread_id": f"{intent}_{i}",
                    "intent": intent,
                    "resolved": True,
                }
            )
    # a handful of unresolved threads that must never land in eval
    for i in range(5):
        rows.append({"thread_id": f"unresolved_{i}", "intent": "billing", "resolved": False})
    return pd.DataFrame(rows)


def test_split_is_disjoint():
    threads = _fixture_threads()
    train_df, eval_df = split_threads(threads, holdout_fraction=0.2, seed=42)
    assert set(train_df["thread_id"]) & set(eval_df["thread_id"]) == set()


def test_unresolved_threads_never_in_eval():
    threads = _fixture_threads()
    train_df, eval_df = split_threads(threads, holdout_fraction=0.2, seed=42)
    assert not eval_df["thread_id"].str.startswith("unresolved").any()
    assert eval_df["resolved"].all()


def test_split_is_stratified_by_intent():
    threads = _fixture_threads(n_per_intent=20)
    train_df, eval_df = split_threads(threads, holdout_fraction=0.2, seed=42)
    for intent in ["billing", "device_help", "software"]:
        assert (eval_df["intent"] == intent).sum() == 4  # 20 * 0.2


def test_split_is_deterministic():
    threads = _fixture_threads()
    train1, eval1 = split_threads(threads, holdout_fraction=0.2, seed=42)
    train2, eval2 = split_threads(threads, holdout_fraction=0.2, seed=42)
    assert set(eval1["thread_id"]) == set(eval2["thread_id"])
