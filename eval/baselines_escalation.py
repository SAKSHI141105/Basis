"""Escalation baselines, mirroring the intent-classification baselines
(PRD 4.4 / TRD 4.1): a trivial floor and a simple non-LLM baseline, run
against the same golden set as the main system.
"""
from __future__ import annotations

from pipeline.escalation import detect_risk_flags


def trivial_always_escalate(messages: list[str]) -> list[str]:
    """Trivial baseline: escalate everything. Establishes the recall=1.0,
    precision=base-rate floor — cheap to beat on precision, hard to beat on
    recall, which is exactly the tension PRD 6 cares about."""
    return ["escalate"] * len(messages)


def simple_keyword_rule(messages: list[str]) -> list[str]:
    """Simple baseline: escalate only on a risk/human-request keyword hit.

    Unlike the main system, this has no intent-confidence or retrieval-
    similarity signal available (those require the LLM classifier and
    retrieval index) — it is deliberately a weaker, keyword-only rule so it's
    a real baseline, not a strawman with the main system's own signals
    smuggled in.
    """
    return ["escalate" if detect_risk_flags(m) else "auto_handle" for m in messages]
