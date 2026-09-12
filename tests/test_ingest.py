from pathlib import Path

from pipeline.ingest import load_raw, reconstruct_threads

FIXTURE = Path(__file__).parent / "fixtures" / "twcs_sample.csv"


def test_reconstruct_threads_basic_shape():
    df = load_raw(FIXTURE)
    threads = reconstruct_threads(df, brand="AppleSupport")

    # tweet 9 (brand replying to brand) must be dropped, leaving 3 real threads
    assert len(threads) == 3
    assert set(threads.columns) >= {
        "thread_id",
        "customer_msg",
        "brand_reply",
        "followup_msg",
        "turn_count",
        "has_followup",
        "resolved",
        "timestamp",
    }


def test_no_followup_thread_is_resolved():
    df = load_raw(FIXTURE)
    threads = reconstruct_threads(df, brand="AppleSupport")
    row = threads[threads["thread_id"] == "1_2"].iloc[0]
    assert row["has_followup"] == False
    assert row["resolved"] == True
    assert row["turn_count"] == 2


def test_closure_signal_marks_resolved():
    df = load_raw(FIXTURE)
    threads = reconstruct_threads(df, brand="AppleSupport")
    row = threads[threads["thread_id"] == "3_4"].iloc[0]
    assert row["has_followup"] == True
    assert "thanks" in row["followup_msg"].lower()
    assert row["resolved"] == True


def test_unresolved_followup_without_closure_signal():
    df = load_raw(FIXTURE)
    threads = reconstruct_threads(df, brand="AppleSupport")
    row = threads[threads["thread_id"] == "6_7"].iloc[0]
    assert row["has_followup"] == True
    assert row["resolved"] == False


def test_brand_replying_to_brand_is_skipped():
    df = load_raw(FIXTURE)
    threads = reconstruct_threads(df, brand="AppleSupport")
    assert not threads["thread_id"].str.startswith("2_9").any()
