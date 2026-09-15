"""Baseline intent-classification metrics on the full, naturally-imbalanced
eval split (as opposed to the deliberately-stratified 198-example golden
set) -- REPORT.md §8's "what's misleading about the headline number" needs
both distributions to make its point, but only the golden-set numbers came
out of `eval.run_eval`. This is the previously-missing, reproducible source
for the other side: no API calls, just the two offline baselines against
`artifacts/threads_eval.parquet`.
"""
from __future__ import annotations

import json

import pandas as pd

from eval.metrics import compute_intent_metrics
from pipeline.baselines import TrivialMajorityClassifier, load_baseline
from pipeline.config import ARTIFACTS_DIR, TRAIN_THREADS_PARQUET, EVAL_THREADS_PARQUET

OUT_PATH = ARTIFACTS_DIR / "full_split_baseline_report.json"


def run() -> dict:
    train_df = pd.read_parquet(TRAIN_THREADS_PARQUET)
    eval_df = pd.read_parquet(EVAL_THREADS_PARQUET)

    y_true = eval_df["intent"].tolist()

    trivial = TrivialMajorityClassifier().fit(train_df["intent"].tolist())
    trivial_pred = trivial.predict(eval_df["customer_msg_clean"].tolist())
    trivial_metrics = compute_intent_metrics(y_true, trivial_pred)

    simple = load_baseline()
    simple_pred = simple.predict(eval_df["customer_msg_clean"].tolist())
    simple_metrics = compute_intent_metrics(y_true, list(simple_pred))

    report = {
        "eval_split_size": len(eval_df),
        "intent_distribution": eval_df["intent"].value_counts().to_dict(),
        "trivial": {"accuracy": trivial_metrics.accuracy, "macro_f1": trivial_metrics.macro_f1},
        "simple": {"accuracy": simple_metrics.accuracy, "macro_f1": simple_metrics.macro_f1},
    }
    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    report = run()
    print(json.dumps(report, indent=2))
