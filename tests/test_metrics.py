from eval.metrics import compute_escalation_metrics, compute_intent_metrics


def test_intent_metrics_perfect_predictions():
    y_true = ["billing", "billing", "device_help", "device_help"]
    y_pred = ["billing", "billing", "device_help", "device_help"]
    metrics = compute_intent_metrics(y_true, y_pred)
    assert metrics.accuracy == 1.0
    assert metrics.macro_f1 == 1.0


def test_intent_metrics_partial_predictions():
    y_true = ["billing", "billing", "device_help", "device_help"]
    y_pred = ["billing", "device_help", "device_help", "device_help"]
    metrics = compute_intent_metrics(y_true, y_pred)
    assert 0.0 < metrics.accuracy < 1.0
    assert metrics.confusion_matrix is not None
    assert "billing" in metrics.labels


def test_escalation_metrics_perfect_predictions():
    y_true = ["escalate", "auto_handle", "escalate", "auto_handle"]
    y_pred = ["escalate", "auto_handle", "escalate", "auto_handle"]
    metrics = compute_escalation_metrics(y_true, y_pred)
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.false_auto_handle_count == 0
    assert metrics.cost_weighted_score == 1.0


def test_escalation_metrics_false_auto_handle_penalized_more_than_false_escalate():
    # one false auto-handle (missed a true escalate)
    y_true_fah = ["escalate", "auto_handle"]
    y_pred_fah = ["auto_handle", "auto_handle"]
    fah_metrics = compute_escalation_metrics(y_true_fah, y_pred_fah)

    # one false escalate (over-escalated a true auto_handle)
    y_true_fe = ["escalate", "auto_handle"]
    y_pred_fe = ["escalate", "escalate"]
    fe_metrics = compute_escalation_metrics(y_true_fe, y_pred_fe)

    assert fah_metrics.cost_weighted_score < fe_metrics.cost_weighted_score


def test_escalation_metrics_counts():
    y_true = ["escalate", "auto_handle", "escalate"]
    y_pred = ["auto_handle", "escalate", "escalate"]
    metrics = compute_escalation_metrics(y_true, y_pred)
    assert metrics.false_auto_handle_count == 1
    assert metrics.false_escalate_count == 1
