"""Lightweight lexicon-based sentiment scoring for the sentiment_delta signal (TRD 6.1).

Deliberately not a model dependency — this only needs to detect a coarse
worsening-frustration trend across turns, not fine-grained sentiment
analysis, so a small hand-rolled lexicon keeps the dependency surface flat.
"""
from __future__ import annotations

import re

POSITIVE_WORDS = {
    "thanks", "thank", "great", "awesome", "perfect", "appreciate", "resolved",
    "solved", "helpful", "good", "excellent", "works", "worked", "happy",
}
NEGATIVE_WORDS = {
    "frustrated", "frustrating", "angry", "annoyed", "annoying", "terrible",
    "awful", "worst", "useless", "ridiculous", "unacceptable", "disappointed",
    "still", "again", "never", "broken", "waste", "horrible", "furious",
}

WORD_RE = re.compile(r"[a-z']+")


def score_message(text: str) -> float:
    """Return a score in [-1, 1]: positive = positive sentiment, negative = negative."""
    words = WORD_RE.findall(text.lower())
    if not words:
        return 0.0
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    if pos == 0 and neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


def sentiment_delta(thread_context: list[str]) -> float:
    """Score the trend across turns: last turn's sentiment minus the first turn's.

    Negative delta = sentiment worsened as the conversation progressed
    (escalating frustration signal, TRD 6.1). Returns 0.0 for <2 turns.
    """
    if len(thread_context) < 2:
        return 0.0
    first_score = score_message(thread_context[0])
    last_score = score_message(thread_context[-1])
    return last_score - first_score
