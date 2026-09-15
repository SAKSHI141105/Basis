import pandas as pd

from pipeline.resolution_quality import add_resolution_quality_notes, resolution_quality_note


def test_no_followup_flags_silence_only():
    row = pd.Series({"has_followup": False, "resolved": True, "followup_msg": None})
    note = resolution_quality_note(row)
    assert "silence only" in note.lower()


def test_followup_with_closure_signal_flags_strongest_evidence():
    row = pd.Series({"has_followup": True, "resolved": True, "followup_msg": "thanks that worked!"})
    note = resolution_quality_note(row)
    assert "closure phrase" in note.lower()
    assert "strongest evidence" in note.lower()


def test_followup_without_closure_but_resolved_flags_time_gap_fallback():
    row = pd.Series({"has_followup": True, "resolved": True, "followup_msg": "still broken honestly"})
    note = resolution_quality_note(row)
    assert "time-gap" in note.lower() or "weak evidence" in note.lower()


def test_followup_without_closure_and_unresolved_flags_active_issue():
    row = pd.Series({"has_followup": True, "resolved": False, "followup_msg": "still broken honestly"})
    note = resolution_quality_note(row)
    assert "unresolved" in note.lower()
    assert "ongoing" in note.lower()


def test_add_resolution_quality_notes_adds_column_to_all_rows():
    df = pd.DataFrame([
        {"has_followup": False, "resolved": True, "followup_msg": None},
        {"has_followup": True, "resolved": True, "followup_msg": "thanks!"},
    ])
    result = add_resolution_quality_notes(df)
    assert "resolution_quality_note" in result.columns
    assert result["resolution_quality_note"].notna().all()
