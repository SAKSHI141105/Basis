"""Merge hand-labels (true_intent, true_escalation, ambiguous_note) collected
via the labeling tool with the sampled golden-set thread data, producing
golden_set.jsonl per TRD 7.1's schema.

Labels are collected out-of-band (a labeling artifact, see labeling_guide.md)
and handed to this module as a plain {thread_id: {true_intent, true_escalation,
ambiguous_note}} JSON mapping — this module has no network access to fetch
them itself.
"""
from __future__ import annotations

import json

import pandas as pd

from pipeline.config import ARTIFACTS_DIR

GOLDEN_SET_RAW_PARQUET = ARTIFACTS_DIR / "golden_set_raw.parquet"
GOLDEN_LABELS_JSON = ARTIFACTS_DIR / "golden_labels.json"
GOLDEN_SET_JSONL = ARTIFACTS_DIR.parent / "golden_set.jsonl"


def merge_labels(raw: pd.DataFrame, labels: dict[str, dict]) -> list[dict]:
    """Return one record per labeled thread (unlabeled threads are dropped —
    the golden set is only as large as what's actually been hand-labeled)."""
    records = []
    raw_by_id = raw.set_index("thread_id", drop=False)
    for thread_id, label in labels.items():
        if thread_id not in raw_by_id.index:
            continue
        if not label.get("true_intent") or not label.get("true_escalation"):
            continue
        row = raw_by_id.loc[thread_id]
        records.append(
            {
                "thread_id": thread_id,
                "customer_msg": row["customer_msg_clean"],
                "brand_reply": row["brand_reply_clean"],
                "turn_count": int(row["turn_count"]),
                "has_followup": bool(row["has_followup"]),
                "suggested_intent": row["intent"],
                "true_intent": label["true_intent"],
                "true_escalation": label["true_escalation"],
                "ambiguous_note": label.get("ambiguous_note") or "",
            }
        )
    return records


def run(
    raw_path=GOLDEN_SET_RAW_PARQUET,
    labels_path=GOLDEN_LABELS_JSON,
    out_path=GOLDEN_SET_JSONL,
) -> list[dict]:
    raw = pd.read_parquet(raw_path)
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    records = merge_labels(raw, labels)
    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records


if __name__ == "__main__":
    records = run()
    print(f"wrote {len(records)} labeled examples to {GOLDEN_SET_JSONL}")
