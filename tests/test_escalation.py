from pipeline.escalation import EscalationSignals, decide, detect_risk_flags


def test_risk_keyword_always_escalates_regardless_of_other_signals():
    signals = EscalationSignals(
        intent_confidence=0.99,
        max_retrieval_similarity=0.99,
        message="I'm going to sue Apple over this stolen data breach",
    )
    result = decide(signals)
    assert result.decision == "escalate"
    assert "safety/legal" in result.reason


def test_explicit_human_request_escalates():
    signals = EscalationSignals(
        intent_confidence=0.95,
        max_retrieval_similarity=0.9,
        message="Please let me speak to a human agent",
    )
    result = decide(signals)
    assert result.decision == "escalate"
    assert "human" in result.reason


def test_low_confidence_escalates():
    signals = EscalationSignals(
        intent_confidence=0.3,
        max_retrieval_similarity=0.9,
        message="my phone is weird",
    )
    result = decide(signals)
    assert result.decision == "escalate"
    assert "intent_confidence" in result.reason


def test_low_retrieval_similarity_escalates():
    signals = EscalationSignals(
        intent_confidence=0.99,
        max_retrieval_similarity=0.2,
        message="my phone is weird",
    )
    result = decide(signals)
    assert result.decision == "escalate"
    assert "retrieval_similarity" in result.reason


def test_repeated_contact_with_worsening_sentiment_escalates():
    signals = EscalationSignals(
        intent_confidence=0.9,
        max_retrieval_similarity=0.9,
        message="still broken",
        contact_count=4,
        sentiment_delta=-0.5,
    )
    result = decide(signals)
    assert result.decision == "escalate"
    assert "repeated contact" in result.reason


def test_confident_and_grounded_auto_handles():
    signals = EscalationSignals(
        intent_confidence=0.99,
        max_retrieval_similarity=0.99,
        message="how do I update my iOS software",
    )
    result = decide(signals)
    assert result.decision == "auto_handle"


def test_moderate_similarity_now_escalates_after_threshold_retuning():
    # Documents the recalibration (DECISION_LOG.md): 0.55 let almost every
    # retrieved precedent through, driving recall down to 0.146 on the real
    # golden set. A retuning pass (twice, now landing at 0.85 similarity as
    # a deliberate, less-aggressive-than-cost-optimal middle ground -- see
    # the long comment in pipeline/escalation.py) still correctly escalates
    # a below-threshold precedent match. High confidence here isolates the
    # similarity check specifically.
    signals = EscalationSignals(
        intent_confidence=0.99,
        max_retrieval_similarity=0.7,
        message="how do I update my iOS software",
    )
    result = decide(signals)
    assert result.decision == "escalate"
    assert "retrieval_similarity" in result.reason
    assert "retrieval_similarity" in result.reason


def test_detect_risk_flags_empty_for_benign_message():
    assert detect_risk_flags("how do I reset my apple id password") == []


def test_spanish_risk_keyword_escalates():
    signals = EscalationSignals(
        intent_confidence=0.99,
        max_retrieval_similarity=0.99,
        message="voy a demandar a Apple, esto es fraude y me han robado dinero",
    )
    result = decide(signals)
    assert result.decision == "escalate"
    assert "safety/legal" in result.reason


def test_hindi_human_request_escalates():
    signals = EscalationSignals(
        intent_confidence=0.95,
        max_retrieval_similarity=0.9,
        message="मुझे किसी इंसान से बात करनी है",
    )
    result = decide(signals)
    assert result.decision == "escalate"
    assert "human" in result.reason
