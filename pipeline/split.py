"""Freeze the train/eval split immediately after cleaning, before any model
or index is built (Architecture §6: golden-eval threads must never leak into
the retrieval index or baseline training data).

Only resolved threads are eligible for the eval holdout, since the eval
split is where the golden set gets sampled from and grounded-reply quality
needs a real resolution to grade against.
"""
from __future__ import annotations

import pandas as pd

from pipeline.config import (
    EVAL_HOLDOUT_FRACTION,
    EVAL_THREADS_PARQUET,
    RANDOM_SEED,
    THREADS_PARQUET,
    TRAIN_THREADS_PARQUET,
)


def split_threads(
    threads: pd.DataFrame,
    holdout_fraction: float = EVAL_HOLDOUT_FRACTION,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into (train, eval), stratified by intent when available.

    Only resolved threads are eligible for eval; unresolved threads always
    go to train (nothing reliable to grade a golden-set answer against).
    """
    eligible = threads[threads["resolved"]]
    ineligible = threads[~threads["resolved"]]

    if "intent" in threads.columns:
        eval_parts = []
        train_parts = [ineligible]
        for _, group in eligible.groupby("intent"):
            eval_group = group.sample(frac=holdout_fraction, random_state=seed)
            eval_parts.append(eval_group)
            train_parts.append(group.drop(eval_group.index))
        eval_df = pd.concat(eval_parts) if eval_parts else eligible.iloc[0:0]
        train_df = pd.concat(train_parts)
    else:
        eval_df = eligible.sample(frac=holdout_fraction, random_state=seed)
        train_df = pd.concat([ineligible, eligible.drop(eval_df.index)])

    return train_df.reset_index(drop=True), eval_df.reset_index(drop=True)


def run(threads: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if threads is None:
        threads = pd.read_parquet(THREADS_PARQUET)

    train_df, eval_df = split_threads(threads)
    TRAIN_THREADS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(TRAIN_THREADS_PARQUET, index=False)
    eval_df.to_parquet(EVAL_THREADS_PARQUET, index=False)
    return train_df, eval_df


if __name__ == "__main__":
    run()
