import pandas as pd

from pipeline.derive_style_guide import measure


def test_measure_computes_expected_stats():
    sample = pd.Series([
        "We're here to help. DM us your details.",
        "Let's look into this together, what iOS version are you on?",
        "Thanks for reaching out.",
    ])
    stats = measure(sample)
    assert stats["n"] == 3
    assert stats["mentions_dm"] == 1
    assert stats["asks_a_question"] == 1
    assert stats["uses_we"] == 1
    assert stats["word_count_min"] <= stats["word_count_mean"] <= stats["word_count_max"]
