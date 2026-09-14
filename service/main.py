"""FastAPI serving layer (Architecture 2.2). Thin — loads frozen artifacts at
startup and exposes /classify, /draft-reply, /decide, and a convenience
/pipeline that chains all three, matching how the demo frontend uses it.
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()  # GEMINI_API_KEY / model names — must run before load_state() reads os.environ

from pipeline.config import ARTIFACTS_DIR, ROOT
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

EVAL_REPORT_PATH = ARTIFACTS_DIR / "eval_report.json"
GOLDEN_SET_PATH = ROOT / "golden_set.jsonl"
SAMPLE_COUNT = 10


def create_app(state: AppState | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.app_state = state if state is not None else load_state()
        yield

    app = FastAPI(title="AppleSupport AI Support Agent", lifespan=lifespan)

    allowed_origins = os.environ.get("CORS_ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/eval-report")
    def eval_report_endpoint() -> dict:
        """Serves the static, committed eval report (Architecture 2.4) --
        no live recompute, so the dashboard always matches the written report."""
        if not EVAL_REPORT_PATH.exists():
            raise HTTPException(
                status_code=404,
                detail="eval_report.json not yet generated -- run `make eval-live` or `make eval-fast`.",
            )
        with open(EVAL_REPORT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    @app.get("/samples")
    def samples_endpoint() -> list[dict]:
        """Real customer messages from the committed golden set, for the demo
        panel's picker (Design Brief 3.2) -- never fabricated examples."""
        if not GOLDEN_SET_PATH.exists():
            raise HTTPException(status_code=404, detail="golden_set.jsonl not found.")
        samples = []
        with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                samples.append({"thread_id": record["thread_id"], "message": record["customer_msg"]})
                if len(samples) >= SAMPLE_COUNT:
                    break
        return samples

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
