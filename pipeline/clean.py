"""Stage 2: clean reconstructed threads for embedding/modeling.

Deliberately light-touch — per TRD 2.2, noise is part of the assignment's
premise, so this strips only threading artifacts and near-duplicates, it
does not normalize away real signal (typos, sentiment, punctuation).
"""
from __future__ import annotations

import re

import pandas as pd

from pipeline.config import THREADS_PARQUET

HANDLE_RE = re.compile(r"@[A-Za-z0-9_]+")
URL_RE = re.compile(r"https?://\S+")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Strip @handles used purely for threading and URLs, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = HANDLE_RE.sub("", text)
    text = URL_RE.sub("", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def dedupe_near_identical(df: pd.DataFrame, subset_col: str = "customer_msg_clean") -> pd.DataFrame:
    """Drop repeated-contact duplicates: same cleaned customer message text.

    Keeps the earliest occurrence (by timestamp) of each duplicate group.
    """
    df = df.sort_values("timestamp")
    return df.drop_duplicates(subset=[subset_col], keep="first")


def drop_no_brand_reply(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["brand_reply"].notna() & (df["brand_reply"].str.len() > 0)]


def run(threads: pd.DataFrame | None = None) -> pd.DataFrame:
    if threads is None:
        threads = pd.read_parquet(THREADS_PARQUET)

    threads = drop_no_brand_reply(threads)
    threads = threads.copy()
    threads["customer_msg_clean"] = threads["customer_msg"].map(clean_text)
    threads["brand_reply_clean"] = threads["brand_reply"].map(clean_text)
    threads = threads[threads["customer_msg_clean"].str.len() > 0]
    threads = dedupe_near_identical(threads)

    threads.to_parquet(THREADS_PARQUET, index=False)
    return threads


if __name__ == "__main__":
    run()
