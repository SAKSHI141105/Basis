"""Grid-search pipeline/escalation.py's thresholds against real golden-set
signals (TRD §6.2's thresholds were "tuned against the golden set,
documented" -- this is that tuning, done once real data existed).

Zero new API calls: replays the actual `decide()` function (so hard
triggers/risk-flag logic stay identical) against the intent_confidence /
max_retrieval_similarity / message already persisted in eval_report.json
by a previous eval run.

Caveat, disclosed not hidden (see DECISION_LOG.md): most true_escalation
labels in the golden set are themselves AI-generated (97%), which showed
weak self-consistency against real human labels on spot-check (3/6 on
escalation specifically). Thresholds tuned here optimize agreement with
that ground truth, which inherits its own reliability question -- this is
a real, documented limitation of the tuning exercise itself, not swept
under the rug.
"""
from __future__ import annotations

import json

from pipeline.config import ARTIFACTS_DIR
from pipeline.escalation import EscalationSignals, decide
from eval.metrics import compute_escalation_metrics

EVAL_REPORT_PATH = ARTIFACTS_DIR / "eval_report.json"

CONFIDENCE_CANDIDATES = [round(x * 0.05, 2) for x in range(1, 20)]  # 0.05 .. 0.95
SIMILARITY_CANDIDATES = [round(x * 0.05, 2) for x in range(1, 20)]


def load_signals(path=EVAL_REPORT_PATH) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        report = json.load(f)
    signals = report.get("escalation_signals")
    if not signals:
        raise RuntimeError(
            f"No escalation_signals in {path} -- re-run `make eval-fast` "
            "with the current eval/run_eval.py to populate them."
        )
    return signals


def evaluate_thresholds(
    signals: list[dict], confidence_threshold: float, similarity_threshold: float
) -> dict:
    preds = []
    for s in signals:
        result = decide(
            EscalationSignals(
                intent_confidence=s["intent_confidence"],
                max_retrieval_similarity=s["max_retrieval_similarity"],
                message=s["message"],
                contact_count=1,
                sentiment_delta=0.0,
            ),
            confidence_threshold=confidence_threshold,
            similarity_threshold=similarity_threshold,
        )
        preds.append(result.decision)
    true = [s["true_escalation"] for s in signals]
    metrics = compute_escalation_metrics(true, preds)
    return metrics.__dict__


def grid_search(
    signals: list[dict],
    confidence_candidates: list[float] = CONFIDENCE_CANDIDATES,
    similarity_candidates: list[float] = SIMILARITY_CANDIDATES,
) -> list[dict]:
    """Returns every (confidence_threshold, similarity_threshold, metrics)
    combination, sorted best-cost_weighted_score first."""
    results = []
    for conf in confidence_candidates:
        for sim in similarity_candidates:
            metrics = evaluate_thresholds(signals, conf, sim)
            results.append(
                {"confidence_threshold": conf, "similarity_threshold": sim, **metrics}
            )
    results.sort(key=lambda r: r["cost_weighted_score"], reverse=True)
    return results


if __name__ == "__main__":
    signals = load_signals()
    results = grid_search(signals)

    current = evaluate_thresholds(signals, 0.6, 0.55)
    print(f"CURRENT (conf=0.60, sim=0.55): cost_weighted={current['cost_weighted_score']:.3f} "
          f"precision={current['precision']:.3f} recall={current['recall']:.3f}")
    print()
    print("Top 10 candidates by cost-weighted score:")
    for r in results[:10]:
        print(
            f"  conf={r['confidence_threshold']:.2f} sim={r['similarity_threshold']:.2f}  "
            f"cost_weighted={r['cost_weighted_score']:.3f}  precision={r['precision']:.3f}  "
            f"recall={r['recall']:.3f}  f1={r['f1']:.3f}"
        )
