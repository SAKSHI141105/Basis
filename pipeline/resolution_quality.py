"""Per-example resolution-quality notes for the golden set (PRD 4.4: golden
set entries need "intent + escalation ground truth + reference resolution
quality notes").

Derived entirely from already-computed thread fields (has_followup,
followup_msg, resolved) -- zero new API calls, zero new labeling pass.
Classifies *how* the weak "resolved" heuristic (TRD 2.3) reached its
verdict for this specific example, since "resolved=True" hides three very
different levels of evidence.
"""
from __future__ import annotations

import pandas as pd

from pipeline.config import CLOSURE_SIGNALS


def _has_closure_signal(text: str) -> bool:
    lowered = text.lower()
    return any(signal in lowered for signal in CLOSURE_SIGNALS)


def resolution_quality_note(row: pd.Series) -> str:
    """One-sentence, honest note on the strength of evidence behind this
    example's `resolved` flag -- not a re-judgment of intent/escalation."""
    has_followup = bool(row.get("has_followup"))
    resolved = bool(row.get("resolved"))
    followup_msg = row.get("followup_msg")

    if not has_followup:
        return (
            "Resolved by silence only (TRD 2.3): the customer never replied "
            "again after the brand's response. This is the weakest form of "
            "evidence this heuristic produces -- silence is not confirmed "
            "satisfaction, just absence of a recorded complaint."
        )

    if pd.notna(followup_msg) and _has_closure_signal(str(followup_msg)):
        return (
            "Resolved by an explicit closure phrase in the customer's own "
            "follow-up (e.g. \"thanks\"/\"got it\") -- the strongest evidence "
            "this heuristic produces, though still self-reported by the "
            "customer, not independently verified."
        )

    if resolved:
        return (
            "Customer did follow up, but with no closure phrase -- marked "
            "resolved only because that follow-up came after the silence "
            "window (TRD 2.3's time-gap fallback). Weak evidence: the "
            "follow-up's actual content doesn't confirm the issue was fixed."
        )

    return (
        "Marked unresolved: the customer's follow-up contains no closure "
        "phrase and arrived within the silence window, i.e. this thread "
        "shows an active, apparently-ongoing issue at the point captured."
    )


def add_resolution_quality_notes(golden_set_raw: pd.DataFrame) -> pd.DataFrame:
    df = golden_set_raw.copy()
    df["resolution_quality_note"] = df.apply(resolution_quality_note, axis=1)
    return df
