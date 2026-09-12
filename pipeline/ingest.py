"""Stage 1: ingest twcs.csv, filter to the brand, reconstruct threads.

Reconstructs customer -> brand-reply -> (optional) customer follow-up chains
from the flat Kaggle "Customer Support on Twitter" export, bounded to
MAX_THREAD_DEPTH turns (TRD 2.1).
"""
from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

from pipeline.config import (
    BRAND,
    CLOSURE_SIGNALS,
    MAX_THREAD_DEPTH,
    RAW_TWCS_CSV,
    RESOLUTION_SILENCE_HOURS,
    THREADS_PARQUET,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TWCS_DTYPES = {
    "tweet_id": "Int64",
    "author_id": "string",
    "inbound": "bool",
    "text": "string",
    "response_tweet_id": "string",
    "in_response_to_tweet_id": "Int64",
}


def load_raw(csv_path=RAW_TWCS_CSV) -> pd.DataFrame:
    """Load the raw Kaggle export.

    `created_at` in the real dataset is a Twitter-format string
    (e.g. "Tue Oct 31 22:10:47 +0000 2017"); parsed permissively so a
    malformed row doesn't fail the whole ingest.
    """
    df = pd.read_csv(csv_path, dtype=TWCS_DTYPES)
    df["created_at"] = pd.to_datetime(
        df["created_at"], format="%a %b %d %H:%M:%S %z %Y", errors="coerce", utc=True
    )
    return df


def _closure_signal_present(text: str) -> bool:
    lowered = text.lower()
    return any(signal in lowered for signal in CLOSURE_SIGNALS)


def reconstruct_threads(
    df: pd.DataFrame,
    brand: str = BRAND,
    max_depth: int = MAX_THREAD_DEPTH,
    silence_hours: int = RESOLUTION_SILENCE_HOURS,
) -> pd.DataFrame:
    """Reconstruct customer -> brand-reply -> follow-up chains for one brand.

    A thread is anchored on a brand reply (`author_id == brand`) that is
    itself a response to an inbound customer tweet. We walk backward via
    `in_response_to_tweet_id` to find the originating customer message, and
    forward via `response_tweet_id` to find any customer follow-up — up to
    `max_depth` total turns, per TRD 2.1.
    """
    by_id = df.set_index("tweet_id", drop=False)
    brand_replies = df[(df["author_id"] == brand) & (~df["inbound"])]

    rows = []
    for _, reply in brand_replies.iterrows():
        parent_id = reply["in_response_to_tweet_id"]
        if pd.isna(parent_id) or parent_id not in by_id.index:
            continue
        customer_msg = by_id.loc[parent_id]
        if not bool(customer_msg["inbound"]):
            continue  # brand replying to itself or another brand account

        turn_count = 2
        has_followup = False
        followup_text = None
        followup_ts = None

        raw_response_ids = reply.get("response_tweet_id")
        followup_ids = "" if pd.isna(raw_response_ids) else str(raw_response_ids)
        for fid_str in followup_ids.split(","):
            fid_str = fid_str.strip()
            if not fid_str or not fid_str.isdigit():
                continue
            fid = int(fid_str)
            if fid not in by_id.index:
                continue
            candidate = by_id.loc[fid]
            if bool(candidate["inbound"]):
                has_followup = True
                followup_text = candidate["text"]
                followup_ts = candidate["created_at"]
                turn_count = 3
                break

        if turn_count > max_depth:
            continue

        resolved_by_silence = True
        if has_followup and pd.notna(followup_ts) and pd.notna(reply["created_at"]):
            hours_gap = (followup_ts - reply["created_at"]).total_seconds() / 3600.0
            resolved_by_silence = hours_gap > silence_hours or hours_gap < 0
        resolved_by_closure = bool(followup_text) and _closure_signal_present(followup_text)
        resolved = (not has_followup) or resolved_by_silence or resolved_by_closure

        rows.append(
            {
                "thread_id": f"{customer_msg['tweet_id']}_{reply['tweet_id']}",
                "customer_msg": customer_msg["text"],
                "brand_reply": reply["text"],
                "followup_msg": followup_text,
                "turn_count": turn_count,
                "has_followup": has_followup,
                "resolved": resolved,
                "timestamp": customer_msg["created_at"],
            }
        )

    threads = pd.DataFrame(rows)
    logger.info("reconstructed %d threads for brand=%s", len(threads), brand)
    return threads


def run(csv_path=RAW_TWCS_CSV, out_path=THREADS_PARQUET) -> pd.DataFrame:
    df = load_raw(csv_path)
    threads = reconstruct_threads(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    threads.to_parquet(out_path, index=False)
    logger.info("wrote %s (%d rows)", out_path, len(threads))
    return threads


if __name__ == "__main__":
    run()
