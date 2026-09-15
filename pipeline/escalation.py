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
        # English
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
        # Spanish
        r"\bdemandar\b",
        r"\babogado\b",
        r"\bacci[oó]n legal\b",
        r"\bsuicid",
        r"\bfraude\b",
        r"\brobado\b",
        r"\bhackeado\b",
        r"\bcobraron dos veces\b",
        r"\bcargo no autorizado\b",
        # Portuguese
        r"\bprocessar\b",
        r"\badvogado\b",
        r"\baç[aã]o legal\b",
        r"\bfraude\b",
        r"\broubado\b",
        r"\bhackeado\b",
        r"\bcobrado(a)? duas vezes\b",
        r"\bcobran[çc]a n[ãa]o autorizada\b",
        # French
        r"\bpoursuivre en justice\b",
        r"\bavocat\b",
        r"\baction en justice\b",
        r"\bfraude\b",
        r"\bvol[ée]\b",
        r"\bpirat[ée]\b",
        r"\bfacturé deux fois\b",
        # German
        r"\bverklagen\b",
        r"\banwalt\b",
        r"\brechtliche schritte\b",
        r"\bbetrug\b",
        r"\bgestohlen\b",
        r"\bgehackt\b",
        r"\bzweimal (belastet|abgebucht)\b",
        # Hindi (Devanagari + common romanized)
        r"मुकदमा",
        r"वकील",
        r"आत्महत्या",
        r"धोखाधड़ी",
        r"चोरी",
        r"हैक",
        r"\bvakeel\b",
        r"\bmukadma\b",
        r"\bdhokhadhadi\b",
    ]
]

HUMAN_REQUEST_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        # English
        r"\btalk to a human\b",
        r"\bspeak to a (person|human|representative|agent)\b",
        r"\breal person\b",
        r"\bcustomer service rep\b",
        r"\bmanager\b",
        # Spanish
        r"\bhablar con una persona\b",
        r"\bhablar con un humano\b",
        r"\bpersona real\b",
        r"\brepresentante de servicio\b",
        r"\bgerente\b",
        # Portuguese
        r"\bfalar com uma pessoa\b",
        r"\bfalar com um humano\b",
        r"\bpessoa real\b",
        r"\brepresentante de atendimento\b",
        r"\bgerente\b",
        # French
        r"\bparler à une personne\b",
        r"\bparler à un humain\b",
        r"\bvrai(e)? personne\b",
        r"\breprésentant du service client\b",
        r"\bresponsable\b",
        # German
        r"\bmit einer person sprechen\b",
        r"\bmit einem menschen sprechen\b",
        r"\becht(er|e)? mensch\b",
        r"\bkundendienstmitarbeiter\b",
        r"\bmanager\b",
        # Hindi (Devanagari + common romanized)
        r"किसी इंसान से बात",
        r"असली व्यक्ति",
        r"मैनेजर",
        r"\bkisi insaan se baat\b",
        r"\basli vyakti\b",
        r"\bmanager se baat\b",
    ]
]

# Best-effort multilingual coverage added when embedding/classification/
# generation were extended beyond PRD's original English-only scope (see
# DECISION_LOG.md). This is NOT an exhaustive per-language taxonomy -- it
# covers the same categories as the English patterns (legal threats,
# self-harm, fraud/theft, account compromise, double-charging, and requests
# for a human) in the handful of languages the golden-set/dataset review
# turned up (Spanish, Portuguese, French, German, Hindi). A message in a
# language/phrasing not covered here still falls through to the confidence/
# similarity threshold checks below, so it is not silently unescalatable --
# just not guaranteed a hard-trigger match.

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
