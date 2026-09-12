"""Automated metrics for intent classification and escalation decisions (TRD 7.2).

Computed identically for trivial baseline, simple baseline, and the main
system against the same golden set, same script, one table — so the
comparison in the report is apples-to-apples, not cherry-picked.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

# Cost weights for the escalation cost-weighted variant (PRD 6): a false
# auto-handle (system said auto_handle, truth was escalate) is a confidently
# wrong reply sent to a customer — weighted worse than an unnecessary
# escalation (system said escalate, truth was auto_handle).
FALSE_AUTO_HANDLE_COST = 3.0
FALSE_ESCALATE_COST = 1.0


@dataclass
class IntentMetrics:
    accuracy: float
    macro_f1: float
    per_intent_f1: dict[str, float]
    confusion_matrix: list[list[int]]
    labels: list[str]


@dataclass
class EscalationMetrics:
    precision: float
    recall: float
    f1: float
    cost_weighted_score: float
    false_auto_handle_count: int
    false_escalate_count: int


def compute_intent_metrics(y_true: list[str], y_pred: list[str]) -> IntentMetrics:
    labels = sorted(set(y_true) | set(y_pred))
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    per_label_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return IntentMetrics(
        accuracy=accuracy,
        macro_f1=float(macro_f1),
        per_intent_f1=dict(zip(labels, [float(x) for x in per_label_f1])),
        confusion_matrix=cm.tolist(),
        labels=labels,
    )


def compute_escalation_metrics(
    y_true: list[str], y_pred: list[str], positive_label: str = "escalate"
) -> EscalationMetrics:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=positive_label, zero_division=0,
    )

    false_auto_handle = sum(
        1 for t, p in zip(y_true, y_pred) if t == positive_label and p != positive_label
    )
    false_escalate = sum(
        1 for t, p in zip(y_true, y_pred) if t != positive_label and p == positive_label
    )
    total_cost = false_auto_handle * FALSE_AUTO_HANDLE_COST + false_escalate * FALSE_ESCALATE_COST
    max_possible_cost = len(y_true) * FALSE_AUTO_HANDLE_COST
    cost_weighted_score = 1.0 - (total_cost / max_possible_cost) if max_possible_cost else 1.0

    return EscalationMetrics(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        cost_weighted_score=cost_weighted_score,
        false_auto_handle_count=false_auto_handle,
        false_escalate_count=false_escalate,
    )
