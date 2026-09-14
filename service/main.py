"""FastAPI serving layer (Architecture 2.2). Thin — loads frozen artifacts at
startup and exposes /classify, /draft-reply, /decide, and a convenience
/pipeline that chains all three, matching how the demo frontend uses it.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request

load_dotenv()  # GEMINI_API_KEY / model names — must run before load_state() reads os.environ

from service.pipeline_runner import run_classify, run_decide, run_draft_reply, run_pipeline
from service.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    DecideRequest,
    DecideResponse,
    DraftReplyRequest,
    DraftReplyResponse,
    PipelineRequest,
    PipelineResponse,
)
from service.state import AppState, load_state


def create_app(state: AppState | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.app_state = state if state is not None else load_state()
        yield

    app = FastAPI(title="AppleSupport AI Support Agent", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/classify", response_model=ClassifyResponse)
    def classify_endpoint(req: ClassifyRequest, request: Request) -> ClassifyResponse:
        result = run_classify(request.app.state.app_state, req.message)
        return ClassifyResponse(
            intent=result.intent, confidence=result.confidence, all_scores=result.all_scores
        )

    @app.post("/draft-reply", response_model=DraftReplyResponse)
    def draft_reply_endpoint(req: DraftReplyRequest, request: Request) -> DraftReplyResponse:
        reply, _ = run_draft_reply(request.app.state.app_state, req.message, req.intent)
        return DraftReplyResponse(
            draft=reply.draft, grounded_on=reply.grounded_on, retrieval_scores=reply.retrieval_scores
        )

    @app.post("/decide", response_model=DecideResponse)
    def decide_endpoint(req: DecideRequest, request: Request) -> DecideResponse:
        result = run_decide(request.app.state.app_state, req.message, req.thread_context)
        return DecideResponse(decision=result.decision, reason=result.reason, signals=result.signals)

    @app.post("/pipeline", response_model=PipelineResponse)
    def pipeline_endpoint(req: PipelineRequest, request: Request) -> PipelineResponse:
        result = run_pipeline(request.app.state.app_state, req.message, req.thread_context)
        classification = result["classify"]
        reply = result["draft_reply"]
        decision = result["decision"]
        return PipelineResponse(
            classify=ClassifyResponse(
                intent=classification.intent,
                confidence=classification.confidence,
                all_scores=classification.all_scores,
            ),
            draft_reply=DraftReplyResponse(
                draft=reply.draft, grounded_on=reply.grounded_on, retrieval_scores=reply.retrieval_scores
            ),
            decision=DecideResponse(
                decision=decision.decision, reason=decision.reason, signals=decision.signals
            ),
        )

    return app


app = create_app()
