import pandas as pd
import pytest

from pipeline.golden_set import sample_golden_set


def _fixture_threads():
    rows = []
    for intent in ["billing", "device_help"]:
        for i in range(30):
            rows.append(
                {
                    "thread_id": f"{intent}_easy_{i}",
                    "intent": intent,
                    "customer_msg_clean": "how do I update my account information please",
                    "turn_count": 2,
                }
            )
        for i in range(10):
            rows.append(
                {
                    "thread_id": f"{intent}_hard_{i}",
                    "intent": intent,
                    "customer_msg_clean": "this is fraud I will sue you",
                    "turn_count": 3,
                }
            )
    return pd.DataFrame(rows)


def test_sample_requires_intent_column():
    with pytest.raises(ValueError):
        sample_golden_set(pd.DataFrame({"customer_msg_clean": ["hi"]}))


def test_sample_respects_target_size_roughly():
    threads = _fixture_threads()
    golden = sample_golden_set(threads, target_size=40)
    assert 30 <= len(golden) <= 40


def test_sample_covers_all_intents():
    threads = _fixture_threads()
    golden = sample_golden_set(threads, target_size=40)
    assert set(golden["intent"].unique()) == {"billing", "device_help"}


def test_sample_includes_hard_cases():
    threads = _fixture_threads()
    golden = sample_golden_set(threads, target_size=40, oversample_fraction=0.3)
    hard_count = golden["thread_id"].str.contains("hard").sum()
    assert hard_count > 0


def test_sample_is_deterministic():
    threads = _fixture_threads()
    g1 = sample_golden_set(threads, target_size=40, seed=42)
    g2 = sample_golden_set(threads, target_size=40, seed=42)
    assert set(g1["thread_id"]) == set(g2["thread_id"])
