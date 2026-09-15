import json

import pandas as pd
import pytest

from eval import full_split_baselines
from pipeline.baselines import save_baseline, train_simple_baseline


@pytest.fixture
def fake_split(tmp_path, monkeypatch):
    train_df = pd.DataFrame(
        {
            "customer_msg_clean": [
                "my phone won't turn on",
                "screen is frozen help",
                "you charged me twice",
                "overcharged for subscription",
            ]
            * 5,
            "intent": ["device_help", "device_help", "billing", "billing"] * 5,
        }
    )
    eval_df = pd.DataFrame(
        {
            "customer_msg_clean": ["phone is dead", "double charged again", "phone is dead"],
            "intent": ["device_help", "billing", "device_help"],
        }
    )
    train_path = tmp_path / "train.parquet"
    eval_path = tmp_path / "eval.parquet"
    train_df.to_parquet(train_path)
    eval_df.to_parquet(eval_path)

    baseline_path = tmp_path / "baseline_model.pkl"
    save_baseline(
        train_simple_baseline(train_df["customer_msg_clean"].tolist(), train_df["intent"].tolist()),
        path=baseline_path,
    )

    out_path = tmp_path / "full_split_baseline_report.json"
    monkeypatch.setattr(full_split_baselines, "TRAIN_THREADS_PARQUET", train_path)
    monkeypatch.setattr(full_split_baselines, "EVAL_THREADS_PARQUET", eval_path)
    monkeypatch.setattr(full_split_baselines, "OUT_PATH", out_path)
    monkeypatch.setattr(
        full_split_baselines, "load_baseline", lambda path=baseline_path: __import__("pickle").load(open(baseline_path, "rb"))
    )
    return out_path


def test_run_writes_report_with_both_baselines(fake_split):
    report = full_split_baselines.run()
    assert report["eval_split_size"] == 3
    assert "trivial" in report and "simple" in report
    assert 0.0 <= report["trivial"]["accuracy"] <= 1.0
    assert 0.0 <= report["simple"]["accuracy"] <= 1.0
    assert json.loads(fake_split.read_text())["eval_split_size"] == 3


def test_trivial_baseline_predicts_majority_train_label(fake_split):
    report = full_split_baselines.run()
    # train split is balanced 50/50, eval split is majority device_help (2/3)
    assert report["intent_distribution"]["device_help"] == 2
