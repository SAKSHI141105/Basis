"""Human-agreement study for the LLM judge (TRD 7.3, mandatory).

Compares judge scores against a human's manual scoring of the same rubric
on a subsample. Reports both Cohen's kappa (categorical: scores bucketed
into low/mid/high) and Spearman correlation (raw 1-5 scores), since kappa
alone can be misleadingly harsh on ordinal data with few categories.
"""
from __future__ import annotations

from dataclasses import dataclass

from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score


def bucket_score(score: float, low_max: float = 2.0, mid_max: float = 3.5) -> str:
    """Bucket a 1-5 score into low/mid/high for categorical kappa agreement."""
    if score <= low_max:
        return "low"
    if score <= mid_max:
        return "mid"
    return "high"


@dataclass
class AgreementReport:
    n: int
    cohen_kappa: float
    spearman_r: float
    spearman_p: float
    mean_absolute_diff: float


def compute_agreement(judge_scores: list[float], human_scores: list[float]) -> AgreementReport:
    if len(judge_scores) != len(human_scores):
        raise ValueError("judge_scores and human_scores must be the same length")
    if len(judge_scores) < 2:
        raise ValueError("need at least 2 examples to compute agreement")

    judge_buckets = [bucket_score(s) for s in judge_scores]
    human_buckets = [bucket_score(s) for s in human_scores]
    kappa = cohen_kappa_score(judge_buckets, human_buckets)

    rho, p_value = spearmanr(judge_scores, human_scores)
    mean_abs_diff = sum(abs(j - h) for j, h in zip(judge_scores, human_scores)) / len(judge_scores)

    return AgreementReport(
        n=len(judge_scores),
        cohen_kappa=float(kappa),
        spearman_r=float(rho),
        spearman_p=float(p_value),
        mean_absolute_diff=mean_abs_diff,
    )
