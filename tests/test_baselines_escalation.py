from eval.baselines_escalation import simple_keyword_rule, trivial_always_escalate


def test_trivial_always_escalate():
    preds = trivial_always_escalate(["hi", "help", "whatever"])
    assert preds == ["escalate", "escalate", "escalate"]


def test_simple_keyword_rule_escalates_on_risk_keyword():
    preds = simple_keyword_rule(["I will sue you", "how do I update my ios"])
    assert preds == ["escalate", "auto_handle"]


def test_simple_keyword_rule_escalates_on_human_request():
    preds = simple_keyword_rule(["let me speak to a human"])
    assert preds == ["escalate"]
