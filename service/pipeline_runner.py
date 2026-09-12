"""Assembles classify -> retrieve -> draft -> decide (Architecture 3), importable
directly by both the FastAPI endpoints and the eval harness (no HTTP round
trip needed for the harness, per Architecture 2.3).
"""
from __future__ import annotations

import numpy as np

from pipeline.classify import ClassificationResult, classify
from pipeline.escalation import EscalationSignals, EscalationResult, decide as escalation_decide
from pipeline.generate import GeneratedReply, generate_reply
from pipeline.index import RetrievalResult
from pipeline.sentiment import sentiment_delta
from service.state import AppState


def run_classify(state: AppState, message: str) -> ClassificationResult:
    return classify(message, state.llm_client, state.classifier_model, taxonomy=state.taxonomy)


def run_retrieve(
    state: AppState, message: str, intent: str | None, k: int = 3
) -> list[RetrievalResult]:
    embedding = np.asarray(state.embed_fn([message])[0])
    return state.retrieval_index.query(embedding, k=k, intent=intent)


def run_draft_reply(
    state: AppState, message: str, intent: str | None = None
) -> tuple[GeneratedReply, list[RetrievalResult]]:
    precedents = run_retrieve(state, message, intent)
    reply = generate_reply(
        message, precedents, state.brand_style_guide, state.llm_client, state.generation_model
    )
    return reply, precedents


def run_decide(
    state: AppState,
    message: str,
    thread_context: list[str] | None = None,
) -> EscalationResult:
    thread_context = thread_context or []
    classification = run_classify(state, message)
    precedents = run_retrieve(state, message, classification.intent)
    max_similarity = max((p.similarity for p in precedents), default=0.0)

    signals = EscalationSignals(
        intent_confidence=classification.confidence,
        max_retrieval_similarity=max_similarity,
        message=message,
        contact_count=len(thread_context),
        sentiment_delta=sentiment_delta(thread_context + [message]) if thread_context else 0.0,
    )
    return escalation_decide(signals)


def run_pipeline(
    state: AppState, message: str, thread_context: list[str] | None = None
) -> dict:
    thread_context = thread_context or []
    classification = run_classify(state, message)
    reply, precedents = run_draft_reply(state, message, classification.intent)
    max_similarity = max((p.similarity for p in precedents), default=0.0)

    signals = EscalationSignals(
        intent_confidence=classification.confidence,
        max_retrieval_similarity=max_similarity,
        message=message,
        contact_count=len(thread_context),
        sentiment_delta=sentiment_delta(thread_context + [message]) if thread_context else 0.0,
    )
    decision = escalation_decide(signals)

    return {"classify": classification, "draft_reply": reply, "decision": decision}
