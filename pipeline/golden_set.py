"""Golden-set sampling (TRD 7.1, PRD 4.4): 150-250 examples, stratified by
derived intent, with deliberate oversampling of low-confidence-looking
messages, multi-turn threads, and escalation-trigger-resembling messages —
so the set isn't dominated by easy cases (PRD risk table, row 3).
"""
from __future__ import annotations

import pandas as pd

from pipeline.config import RANDOM_SEED
from pipeline.escalation import detect_risk_flags

DEFAULT_TARGET_SIZE = 200
OVERSAMPLE_FRACTION = 0.3  # fraction of the golden set reserved for "hard" cases


def _is_hard_case(row: pd.Series) -> bool:
    """A message is a "hard case" if it's multi-turn, resembles an escalation
    trigger, or is unusually short (a proxy for low classifier confidence
    before the classifier even exists to score it)."""
    is_multiturn = row.get("turn_count", 1) > 2
    has_risk_signal = len(detect_risk_flags(row.get("customer_msg_clean", ""))) > 0
    is_short = len(str(row.get("customer_msg_clean", "")).split()) <= 4
    return bool(is_multiturn or has_risk_signal or is_short)


def sample_golden_set(
    threads: pd.DataFrame,
    target_size: int = DEFAULT_TARGET_SIZE,
    oversample_fraction: float = OVERSAMPLE_FRACTION,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Stratified-by-intent sample with a documented hard-case oversample.

    Within each intent's allocation, hard cases (per _is_hard_case) are
    sampled at `oversample_fraction` of that intent's quota when enough
    exist, with the remainder filled by easy cases — falls back to
    whatever's available if an intent doesn't have enough hard cases.
    """
    if "intent" not in threads.columns:
        raise ValueError("threads must have an 'intent' column (apply taxonomy labels first)")

    threads = threads.copy()
    threads["_is_hard"] = threads.apply(_is_hard_case, axis=1)

    intents = threads["intent"].unique()
    per_intent_quota = max(1, target_size // len(intents))

    sampled_parts = []
    for intent in intents:
        group = threads[threads["intent"] == intent]
        quota = min(per_intent_quota, len(group))
        hard_quota = min(int(quota * oversample_fraction), len(group[group["_is_hard"]]))
        easy_quota = quota - hard_quota

        hard_sample = group[group["_is_hard"]].sample(n=hard_quota, random_state=seed) if hard_quota else group.iloc[0:0]
        remaining = group.drop(hard_sample.index)
        easy_pool = remaining[~remaining["_is_hard"]] if len(remaining[~remaining["_is_hard"]]) >= easy_quota else remaining
        easy_sample = easy_pool.sample(n=min(easy_quota, len(easy_pool)), random_state=seed)

        sampled_parts.append(pd.concat([hard_sample, easy_sample]))

    golden = pd.concat(sampled_parts).drop(columns=["_is_hard"]).reset_index(drop=True)
    return golden
