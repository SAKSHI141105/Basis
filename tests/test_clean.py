import pandas as pd

from pipeline.clean import clean_text, dedupe_near_identical, drop_no_brand_reply


def test_clean_text_strips_handles_urls_whitespace():
    raw = "@AppleSupport  please help   check http://apple.com/status now"
    cleaned = clean_text(raw)
    assert "@AppleSupport" not in cleaned
    assert "http://apple.com/status" not in cleaned
    assert "  " not in cleaned


def test_clean_text_idempotent():
    raw = "@user my iPhone is broken https://t.co/xyz plz help"
    once = clean_text(raw)
    twice = clean_text(once)
    assert once == twice


def test_clean_text_handles_non_string():
    assert clean_text(None) == ""


def test_drop_no_brand_reply():
    df = pd.DataFrame({"brand_reply": ["hi", None, ""]})
    result = drop_no_brand_reply(df)
    assert len(result) == 1


def test_dedupe_near_identical_keeps_earliest():
    df = pd.DataFrame(
        {
            "customer_msg_clean": ["help me", "help me", "different"],
            "timestamp": pd.to_datetime(["2018-01-02", "2018-01-01", "2018-01-01"]),
        }
    )
    result = dedupe_near_identical(df)
    assert len(result) == 2
    kept = result[result["customer_msg_clean"] == "help me"].iloc[0]
    assert kept["timestamp"] == pd.Timestamp("2018-01-01")
