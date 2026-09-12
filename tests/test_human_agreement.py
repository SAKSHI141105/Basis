import pytest

from eval.human_agreement import bucket_score, compute_agreement


def test_bucket_score_boundaries():
    assert bucket_score(1.0) == "low"
    assert bucket_score(2.0) == "low"
    assert bucket_score(3.0) == "mid"
    assert bucket_score(5.0) == "high"


def test_perfect_agreement():
    judge = [5.0, 4.0, 3.0, 2.0, 1.0]
    human = [5.0, 4.0, 3.0, 2.0, 1.0]
    report = compute_agreement(judge, human)
    assert report.cohen_kappa == 1.0
    assert report.spearman_r == pytest.approx(1.0)
    assert report.mean_absolute_diff == 0.0


def test_weak_agreement_flagged_by_low_kappa():
    judge = [5.0, 5.0, 5.0, 5.0, 5.0]
    human = [1.0, 2.0, 3.0, 4.0, 5.0]
    report = compute_agreement(judge, human)
    assert report.cohen_kappa < 0.5


def test_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        compute_agreement([1.0, 2.0], [1.0])


def test_too_few_examples_raises():
    with pytest.raises(ValueError):
        compute_agreement([1.0], [1.0])
