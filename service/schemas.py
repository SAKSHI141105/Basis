"""Pydantic request/response contracts, matching TRD 4.3/5.4/6.3 exactly."""
from __future__ import annotations

from pydantic import BaseModel


class ClassifyRequest(BaseModel):
    message: str


class ClassifyResponse(BaseModel):
    intent: str
    confidence: float
    all_scores: dict[str, float]


class DraftReplyRequest(BaseModel):
    message: str
    intent: str | None = None


class DraftReplyResponse(BaseModel):
    draft: str
    grounded_on: list[str]
    retrieval_scores: list[float]


class DecideRequest(BaseModel):
    message: str
    thread_context: list[str] | None = None


class DecideResponse(BaseModel):
    decision: str
    reason: str
    signals: dict


class PipelineRequest(BaseModel):
    message: str
    thread_context: list[str] | None = None


class PipelineResponse(BaseModel):
    classify: ClassifyResponse
    draft_reply: DraftReplyResponse
    decision: DecideResponse
