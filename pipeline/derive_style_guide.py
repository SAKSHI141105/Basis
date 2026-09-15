"""Derive the brand style guide from real, resolved AppleSupport replies
(TRD 5.3: "a short brand style guide, derived by summarizing ~20 real
AppleSupport replies -- tone, length, structure, sign-off pattern"), rather
than writing one by assumption. Prints the sample + measured stats so the
derivation is inspectable, not just asserted.
"""
from __future__ import annotations

import pandas as pd

from pipeline.config import RANDOM_SEED, TRAIN_THREADS_PARQUET

SAMPLE_SIZE = 20


def sample_real_replies(path=TRAIN_THREADS_PARQUET, n: int = SAMPLE_SIZE) -> pd.Series:
    df = pd.read_parquet(path)
    resolved = df[df["resolved"]]
    return resolved["brand_reply_clean"].sample(n, random_state=RANDOM_SEED)


def measure(sample: pd.Series) -> dict:
    lengths = sample.str.split().str.len()
    return {
        "n": len(sample),
        "word_count_min": int(lengths.min()),
        "word_count_max": int(lengths.max()),
        "word_count_mean": round(float(lengths.mean()), 1),
        "mentions_dm": int(sample.str.contains("DM", case=False).sum()),
        "asks_a_question": int(sample.str.contains(r"\?").sum()),
        "uses_we": int(sample.str.contains(r"\bwe\b", case=False).sum()),
    }


if __name__ == "__main__":
    sample = sample_real_replies()
    stats = measure(sample)
    print(f"Sampled {stats['n']} real resolved AppleSupport replies (seed={RANDOM_SEED}):\n")
    for reply in sample:
        print(f"- {reply}")
    print("\nMeasured patterns:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
