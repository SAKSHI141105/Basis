from pipeline.sentiment import score_message, sentiment_delta


def test_score_message_positive():
    assert score_message("thanks so much, that worked perfectly") > 0


def test_score_message_negative():
    assert score_message("this is still broken and it's ridiculous") < 0


def test_score_message_neutral_for_no_lexicon_hits():
    assert score_message("my ipad model number is A1234") == 0.0


def test_sentiment_delta_detects_worsening_trend():
    thread = ["hi, my phone froze", "still broken, this is ridiculous and unacceptable"]
    assert sentiment_delta(thread) < 0


def test_sentiment_delta_detects_improving_trend():
    thread = ["this is frustrating and broken", "thanks, that worked perfectly"]
    assert sentiment_delta(thread) > 0


def test_sentiment_delta_zero_for_single_turn():
    assert sentiment_delta(["just one message"]) == 0.0
