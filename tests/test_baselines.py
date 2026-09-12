import pytest

from pipeline.baselines import TrivialMajorityClassifier, train_simple_baseline

TRAIN_MESSAGES = [
    "my iphone screen is cracked and won't turn on",
    "the screen on my phone is broken and black",
    "how do I update ios on my ipad",
    "what's the process to update to the latest ios version",
    "I was charged twice for my apple music subscription",
    "billing shows a duplicate charge on my account this month",
]
TRAIN_LABELS = [
    "device_troubleshooting",
    "device_troubleshooting",
    "software_update",
    "software_update",
    "billing",
    "billing",
]


def test_trivial_majority_classifier_predicts_constant_label():
    clf = TrivialMajorityClassifier().fit(["a", "a", "b"])
    preds = clf.predict(["msg1", "msg2", "msg3"])
    assert preds == ["a", "a", "a"]


def test_trivial_majority_classifier_requires_fit():
    clf = TrivialMajorityClassifier()
    with pytest.raises(RuntimeError):
        clf.predict(["x"])


def test_trivial_majority_classifier_rejects_empty_labels():
    with pytest.raises(ValueError):
        TrivialMajorityClassifier().fit([])


def test_simple_baseline_learns_separable_classes():
    pipeline = train_simple_baseline(TRAIN_MESSAGES, TRAIN_LABELS)
    preds = pipeline.predict(
        [
            "my screen cracked and the phone won't power on",
            "duplicate billing charge on my card",
        ]
    )
    assert preds[0] == "device_troubleshooting"
    assert preds[1] == "billing"
