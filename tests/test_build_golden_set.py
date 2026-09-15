import pandas as pd

from pipeline.build_golden_set import merge_labels


def _fixture_raw():
    return pd.DataFrame(
        {
            "thread_id": ["a", "b", "c"],
            "customer_msg_clean": ["msg a", "msg b", "msg c"],
            "brand_reply_clean": ["reply a", "reply b", "reply c"],
            "turn_count": [2, 3, 2],
            "has_followup": [False, True, False],
            "intent": ["billing", "device_help", "billing"],
        }
    )


def test_merge_labels_only_includes_fully_labeled():
    raw = _fixture_raw()
    labels = {
        "a": {"true_intent": "billing", "true_escalation": "auto_handle"},
        "b": {"true_intent": "device_help"},  # missing escalation, should be dropped
    }
    records = merge_labels(raw, labels)
    assert len(records) == 1
    assert records[0]["thread_id"] == "a"


def test_merge_labels_includes_resolution_quality_note():
    raw = _fixture_raw()
    labels = {"a": {"true_intent": "billing", "true_escalation": "auto_handle"}}
    records = merge_labels(raw, labels)
    assert "resolution_quality_note" in records[0]
    assert isinstance(records[0]["resolution_quality_note"], str)
    assert len(records[0]["resolution_quality_note"]) > 0


def test_merge_labels_carries_ambiguous_note():
    raw = _fixture_raw()
    labels = {
        "a": {"true_intent": "billing", "true_escalation": "escalate", "ambiguous_note": "borderline case"},
    }
    records = merge_labels(raw, labels)
    assert records[0]["ambiguous_note"] == "borderline case"


def test_merge_labels_ignores_unknown_thread_id():
    raw = _fixture_raw()
    labels = {"nonexistent": {"true_intent": "billing", "true_escalation": "auto_handle"}}
    records = merge_labels(raw, labels)
    assert records == []


def test_merge_labels_default_empty_note():
    raw = _fixture_raw()
    labels = {"c": {"true_intent": "billing", "true_escalation": "auto_handle"}}
    records = merge_labels(raw, labels)
    assert records[0]["ambiguous_note"] == ""
