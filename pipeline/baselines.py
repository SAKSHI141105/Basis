"""Classification baselines (TRD 4.1) — both required, both must actually run.

Trivial baseline: majority-class predictor, establishes the floor.
Simple baseline: TF-IDF (1-2 grams) + Logistic Regression, trained on a
labeled training split distinct from the golden eval set.
"""
from __future__ import annotations

import pickle
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from pipeline.config import ARTIFACTS_DIR, RANDOM_SEED

BASELINE_MODEL_PATH = ARTIFACTS_DIR / "baseline_model.pkl"
TRIVIAL_MAJORITY_PATH = ARTIFACTS_DIR / "trivial_majority_label.txt"


class TrivialMajorityClassifier:
    """Predicts the single majority-class label for every input."""

    def __init__(self):
        self.majority_label: str | None = None

    def fit(self, labels: list[str]) -> "TrivialMajorityClassifier":
        if not labels:
            raise ValueError("cannot fit on empty label list")
        self.majority_label = Counter(labels).most_common(1)[0][0]
        return self

    def predict(self, messages: list[str]) -> list[str]:
        if self.majority_label is None:
            raise RuntimeError("call fit() before predict()")
        return [self.majority_label] * len(messages)


def build_tfidf_logreg_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000)),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced"
                ),
            ),
        ]
    )


def train_simple_baseline(messages: list[str], labels: list[str]) -> Pipeline:
    pipeline = build_tfidf_logreg_pipeline()
    pipeline.fit(messages, labels)
    return pipeline


def save_baseline(pipeline: Pipeline, path=BASELINE_MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)


def load_baseline(path=BASELINE_MODEL_PATH) -> Pipeline:
    with open(path, "rb") as f:
        return pickle.load(f)


def run(train_messages: list[str], train_labels: list[str]) -> None:
    trivial = TrivialMajorityClassifier().fit(train_labels)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    TRIVIAL_MAJORITY_PATH.write_text(trivial.majority_label, encoding="utf-8")

    simple = train_simple_baseline(train_messages, train_labels)
    save_baseline(simple)


if __name__ == "__main__":
    import pandas as pd

    from pipeline.config import TRAIN_THREADS_PARQUET

    df = pd.read_parquet(TRAIN_THREADS_PARQUET)
    run(df["customer_msg_clean"].tolist(), df["intent"].tolist())
