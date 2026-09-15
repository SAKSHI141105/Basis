"""Escalation decision engine (TRD §6).

Deterministic rule layer first (hard triggers always escalate), then a
threshold combination over intent_confidence and max_retrieval_similarity.
Reason strings are templated, not free-form LLM prose, so they stay
auditable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

RISK_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bsue\b",
        r"\blawyer\b",
        r"\blegal action\b",
        r"\bkill myself\b",
        r"\bsuicid",
        r"\bself[\s-]?harm\b",
        r"\bfraud\b",
        r"\bstolen\b",
        r"\bhacked\b",
        r"\bdata breach\b",
        r"\bcharge(d)? me twice\b",
        r"\bunauthorized charge\b",
    ]
]

HUMAN_REQUEST_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\btalk to a human\b",
        r"\bspeak to a (person|human|representative|agent)\b",
        r"\breal person\b",
        r"\bcustomer service rep\b",
        r"\bmanager\b",
    ]
]

# Tuned via grid search against the real golden set once it existed
# (pipeline/tune_escalation_thresholds.py) -- the original 0.6/0.55 defaults
# were reasonable placeholders, never calibrated against labeled data, and
# left escalation recall at 0.146 (missing ~6 of every 7 cases that should
# have escalated). 0.90 similarity roughly quadrupled recall (0.146 -> 0.573)
# while also improving precision (0.480 -> 0.566) -- confidence_threshold
# turned out to barely matter once similarity is this strict, so 0.70 is a
# representative pick from a wide flat region of similarly-good candidates,
# not a sharply-optimal value. Caveat, not hidden: most true_escalation
# labels used to tune this are themselves AI-generated (see DECISION_LOG.md),
# so this tuning optimizes agreement with a ground truth of its own
# uncertain reliability -- a real, not fully human-verified, improvement.
DEFAULT_CONFIDENCE_THRESHOLD = 0.70
DEFAULT_SIMILARITY_THRESHOLD = 0.90
DEFAULT_CONTACT_COUNT_THRESHOLD = 3


@dataclass
class EscalationSignals:
    intent_confidence: float
    max_retrieval_similarity: float
    message: str
    contact_count: int = 0
    sentiment_delta: float = 0.0  # negative = worsening sentiment across turns
    risk_flags: list[str] = field(default_factory=list)


@dataclass
class EscalationResult:
    decision: str  # "auto_handle" | "escalate"
    reason: str
    signals: dict


def detect_risk_flags(message: str) -> list[str]:
    flags = []
    for pattern in RISK_PATTERNS:
        if pattern.search(message):
            flags.append(f"risk_keyword:{pattern.pattern}")
    for pattern in HUMAN_REQUEST_PATTERNS:
        if pattern.search(message):
            flags.append(f"human_request:{pattern.pattern}")
    return flags


def decide(
    signals: EscalationSignals,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    contact_count_threshold: int = DEFAULT_CONTACT_COUNT_THRESHOLD,
) -> EscalationResult:
    risk_flags = signals.risk_flags or detect_risk_flags(signals.message)

    signal_dict = {
        "intent_confidence": signals.intent_confidence,
        "max_retrieval_similarity": signals.max_retrieval_similarity,
        "contact_count": signals.contact_count,
        "sentiment_delta": signals.sentiment_delta,
        "risk_flags": risk_flags,
    }

    # Hard triggers — always escalate, regardless of other signals (TRD §6.2).
    human_requests = [f for f in risk_flags if f.startswith("human_request:")]
    safety_flags = [f for f in risk_flags if f.startswith("risk_keyword:")]

    if safety_flags:
        return EscalationResult(
            decision="escalate",
            reason=f"escalated: safety/legal/financial-harm language detected ({len(safety_flags)} flag(s))",
            signals=signal_dict,
        )
    if human_requests:
        return EscalationResult(
            decision="escalate",
            reason="escalated: explicit request to speak with a human",
            signals=signal_dict,
        )
    if signals.contact_count >= contact_count_threshold and signals.sentiment_delta < 0:
        return EscalationResult(
            decision="escalate",
            reason=(
                f"escalated: repeated contact (contact_count={signals.contact_count} >= "
                f"{contact_count_threshold}) with worsening sentiment (delta={signals.sentiment_delta:.2f})"
            ),
            signals=signal_dict,
        )

    if signals.intent_confidence < confidence_threshold:
        return EscalationResult(
            decision="escalate",
            reason=(
                f"escalated: intent_confidence={signals.intent_confidence:.2f} below threshold "
                f"{confidence_threshold}; classifier is not sure enough"
            ),
            signals=signal_dict,
        )
    if signals.max_retrieval_similarity < similarity_threshold:
        return EscalationResult(
            decision="escalate",
            reason=(
                f"escalated: retrieval_similarity={signals.max_retrieval_similarity:.2f} below threshold "
                f"{similarity_threshold}; no strong precedent found"
            ),
            signals=signal_dict,
        )

    return EscalationResult(
        decision="auto_handle",
        reason=(
            f"auto_handle: intent_confidence={signals.intent_confidence:.2f} >= {confidence_threshold} "
            f"and retrieval_similarity={signals.max_retrieval_similarity:.2f} >= {similarity_threshold}"
        ),
        signals=signal_dict,
    )
