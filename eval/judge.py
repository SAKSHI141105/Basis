"""LLM-as-judge for reply quality (TRD 7.3).

Scores four rubric dimensions (1-5 each): groundedness, correctness/
helpfulness, tone/brand-fit, actionability. The retrieved precedents are
included in the judge prompt so groundedness is checkable against the
actual precedent text, not vibes-based.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from pipeline.index import RetrievalResult
from pipeline.llm_client import LLMClient

JUDGE_TEMPERATURE = 0.0
RUBRIC_DIMENSIONS = ("groundedness", "correctness", "tone", "actionability")


@dataclass
class JudgeScore:
    groundedness: int
    correctness: int
    tone: int
    actionability: int
    rationale: str

    @property
    def mean_score(self) -> float:
        return (self.groundedness + self.correctness + self.tone + self.actionability) / 4.0


def build_judge_prompt(
    message: str, draft: str, precedents: list[RetrievalResult]
) -> str:
    precedent_block = "\n".join(
        f'- [thread_id={p.thread_id}] "{p.brand_reply_clean}"' for p in precedents
    )
    return f"""You are an impartial evaluator scoring a draft customer-support reply.

Customer message: "{message}"

Retrieved precedent replies the draft was supposed to be grounded on:
{precedent_block}

Draft reply to evaluate: "{draft}"

Score the draft 1-5 on each dimension:
- groundedness: does the draft actually reflect the precedents, or hallucinate beyond them?
- correctness: is the draft factually correct and does it actually help?
- tone: does it match a warm-but-concise brand tone?
- actionability: does it give the customer a concrete next step?

Respond with ONLY a JSON object:
{{"groundedness": <1-5>, "correctness": <1-5>, "tone": <1-5>, "actionability": <1-5>, "rationale": "<one sentence>"}}
No other text."""


def _parse_response(raw: str) -> JudgeScore:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    data = json.loads(text)
    return JudgeScore(
        groundedness=int(data["groundedness"]),
        correctness=int(data["correctness"]),
        tone=int(data["tone"]),
        actionability=int(data["actionability"]),
        rationale=data.get("rationale", ""),
    )


def judge_reply(
    message: str,
    draft: str,
    precedents: list[RetrievalResult],
    client: LLMClient,
    model: str,
) -> JudgeScore:
    prompt = build_judge_prompt(message, draft, precedents)
    raw = client.generate(model, prompt, params={"temperature": JUDGE_TEMPERATURE})
    return _parse_response(raw)
