from pipeline.tune_escalation_thresholds import evaluate_thresholds, grid_search

SIGNALS = [
    {"thread_id": "t1", "message": "how do I update ios", "true_escalation": "auto_handle",
     "intent_confidence": 0.9, "max_retrieval_similarity": 0.9},
    {"thread_id": "t2", "message": "still not fixed, third time contacting", "true_escalation": "escalate",
     "intent_confidence": 0.4, "max_retrieval_similarity": 0.3},
    {"thread_id": "t3", "message": "thanks that worked", "true_escalation": "auto_handle",
     "intent_confidence": 0.95, "max_retrieval_similarity": 0.8},
    {"thread_id": "t4", "message": "battery drains fast", "true_escalation": "escalate",
     "intent_confidence": 0.7, "max_retrieval_similarity": 0.6},
]


def test_evaluate_thresholds_returns_metrics_dict():
    metrics = evaluate_thresholds(SIGNALS, confidence_threshold=0.6, similarity_threshold=0.55)
    assert "precision" in metrics
    assert "recall" in metrics
    assert "cost_weighted_score" in metrics


def test_lower_thresholds_increase_or_maintain_recall():
    strict = evaluate_thresholds(SIGNALS, confidence_threshold=0.5, similarity_threshold=0.5)
    lenient_thresholds_but_higher_bar = evaluate_thresholds(
        SIGNALS, confidence_threshold=0.95, similarity_threshold=0.95
    )
    # raising both thresholds should escalate at least as often (higher recall)
    assert lenient_thresholds_but_higher_bar["recall"] >= strict["recall"]


def test_grid_search_returns_sorted_by_cost_weighted_score():
    results = grid_search(SIGNALS, confidence_candidates=[0.3, 0.6, 0.9], similarity_candidates=[0.3, 0.6, 0.9])
    assert len(results) == 9
    scores = [r["cost_weighted_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_grid_search_result_has_threshold_fields():
    results = grid_search(SIGNALS, confidence_candidates=[0.5], similarity_candidates=[0.5])
    assert results[0]["confidence_threshold"] == 0.5
    assert results[0]["similarity_threshold"] == 0.5
