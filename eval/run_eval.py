"""Eval harness entrypoint (TRD 7.4): baselines -> main system -> metrics
table -> judge scoring -> human-agreement report, writing eval_report.json
+ eval_report.md.

Imports the service's core functions directly rather than over HTTP
(Architecture 2.3) so a full run doesn't need the API server up.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from eval.baselines_escalation import simple_keyword_rule, trivial_always_escalate
from eval.human_agreement import compute_agreement
from eval.judge import judge_reply
from eval.metrics import compute_escalation_metrics, compute_intent_metrics
from pipeline.baselines import TRIVIAL_MAJORITY_PATH, load_baseline
from pipeline.build_golden_set import GOLDEN_SET_JSONL
from pipeline.config import ARTIFACTS_DIR
from pipeline.escalation import EscalationSignals, decide as escalation_decide
from service.pipeline_runner import run_classify, run_draft_reply
from service.state import load_state

REPORT_JSON = ARTIFACTS_DIR / "eval_report.json"
REPORT_MD = ARTIFACTS_DIR.parent / "eval_report.md"
HUMAN_JUDGE_SCORES_PATH = ARTIFACTS_DIR / "human_judge_scores.json"


def load_golden_set(path=GOLDEN_SET_JSONL) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _main_system_escalation_decision(state, message: str) -> tuple[str, dict]:
    classification = run_classify(state, message)
    reply, precedents = run_draft_reply(state, message, classification.intent)
    max_similarity = max((p.similarity for p in precedents), default=0.0)
    signals = EscalationSignals(
        intent_confidence=classification.confidence,
        max_retrieval_similarity=max_similarity,
        message=message,
        contact_count=1,
        sentiment_delta=0.0,
    )
    result = escalation_decide(signals)
    return result.decision, {
        "classification": classification,
        "reply": reply,
        "precedents": precedents,
        "escalation_reason": result.reason,
    }


def run(
    golden_set: list[dict] | None = None,
    state=None,
    simple_model=None,
    trivial_label: str | None = None,
    human_scores_path: Path | None = None,
    report_json_path: Path | None = None,
    report_md_path: Path | None = None,
) -> dict:
    load_dotenv()
    golden_set = golden_set if golden_set is not None else load_golden_set()
    if not golden_set:
        raise RuntimeError(
            f"No labeled golden-set examples found at {GOLDEN_SET_JSONL}. "
            "Run `python -m pipeline.build_golden_set` after labeling."
        )

    messages = [ex["customer_msg"] for ex in golden_set]
    true_intents = [ex["true_intent"] for ex in golden_set]
    true_escalations = [ex["true_escalation"] for ex in golden_set]

    # --- Intent: trivial + simple baselines ---
    trivial_label = trivial_label or TRIVIAL_MAJORITY_PATH.read_text(encoding="utf-8").strip()
    trivial_intent_preds = [trivial_label] * len(messages)

    simple_model = simple_model if simple_model is not None else load_baseline()
    simple_intent_preds = list(simple_model.predict(messages))

    # --- Escalation: trivial + simple baselines ---
    trivial_escalation_preds = trivial_always_escalate(messages)
    simple_escalation_preds = simple_keyword_rule(messages)

    # --- Main system: classify + retrieve + generate + decide, per example ---
    state = state if state is not None else load_state()
    main_intent_preds = []
    main_escalation_preds = []
    per_example = []
    for ex, message in zip(golden_set, messages):
        decision, info = _main_system_escalation_decision(state, message)
        main_intent_preds.append(info["classification"].intent)
        main_escalation_preds.append(decision)
        per_example.append({"example": ex, **info})

    intent_metrics = {
        "trivial": compute_intent_metrics(true_intents, trivial_intent_preds).__dict__,
        "simple": compute_intent_metrics(true_intents, simple_intent_preds).__dict__,
        "main": compute_intent_metrics(true_intents, main_intent_preds).__dict__,
    }
    escalation_metrics = {
        "trivial": compute_escalation_metrics(true_escalations, trivial_escalation_preds).__dict__,
        "simple": compute_escalation_metrics(true_escalations, simple_escalation_preds).__dict__,
        "main": compute_escalation_metrics(true_escalations, main_escalation_preds).__dict__,
    }

    # --- LLM judge on the main system's generated replies ---
    judge_scores = []
    for item in per_example:
        score = judge_reply(
            item["example"]["customer_msg"],
            item["reply"].draft,
            item["precedents"],
            state.llm_client,
            os.environ.get("JUDGE_MODEL", "gemini-3.5-flash"),
        )
        judge_scores.append({"thread_id": item["example"]["thread_id"], **score.__dict__, "mean_score": score.mean_score})

    judge_summary = {}
    if judge_scores:
        for dim in ("groundedness", "correctness", "tone", "actionability"):
            judge_summary[f"mean_{dim}"] = sum(s[dim] for s in judge_scores) / len(judge_scores)
        judge_summary["mean_overall"] = sum(s["mean_score"] for s in judge_scores) / len(judge_scores)

    # --- Human-agreement study (optional: only if human scores are provided) ---
    human_agreement = None
    human_scores_path = human_scores_path or HUMAN_JUDGE_SCORES_PATH
    if human_scores_path.exists():
        with open(human_scores_path, "r", encoding="utf-8") as f:
            human_scores_by_id = json.load(f)
        paired_judge, paired_human = [], []
        for s in judge_scores:
            if s["thread_id"] in human_scores_by_id:
                paired_judge.append(s["mean_score"])
                paired_human.append(human_scores_by_id[s["thread_id"]])
        if len(paired_judge) >= 2:
            human_agreement = compute_agreement(paired_judge, paired_human).__dict__

    # --- Failure examples: main system got intent OR escalation wrong ---
    failures = []
    for ex, true_intent, true_esc, pred_intent, pred_esc in zip(
        golden_set, true_intents, true_escalations, main_intent_preds, main_escalation_preds
    ):
        if pred_intent != true_intent or pred_esc != true_esc:
            failures.append(
                {
                    "thread_id": ex["thread_id"],
                    "customer_msg": ex["customer_msg"],
                    "true_intent": true_intent,
                    "pred_intent": pred_intent,
                    "true_escalation": true_esc,
                    "pred_escalation": pred_esc,
                }
            )

    report = {
        "golden_set_size": len(golden_set),
        "intent_metrics": intent_metrics,
        "escalation_metrics": escalation_metrics,
        "judge_scores": judge_scores,
        "judge_summary": judge_summary,
        "human_agreement": human_agreement,
        "failure_examples": failures[:5],
        "misleading_number_note": (
            "What's misleading about the headline accuracy number: it is dominated "
            "by the out_of_scope majority class "
            "(~88% of real traffic per taxonomy.yaml) -- a baseline that always "
            "predicts out_of_scope can beat a real classifier on accuracy while "
            "scoring near-zero on macro-F1. Macro-F1 and per-intent F1 are the "
            "numbers that actually reflect classification quality here."
        ),
    }

    report_json_path = report_json_path or REPORT_JSON
    report_md_path = report_md_path or REPORT_MD
    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    render_markdown(report, report_md_path)
    return report


def render_markdown(report: dict, out_path: Path) -> None:
    lines = ["# Evaluation Report\n"]
    lines.append(f"Golden set size: **{report['golden_set_size']}**\n")

    lines.append("## Intent classification\n")
    lines.append("| System | Accuracy | Macro-F1 |")
    lines.append("|---|---|---|")
    for name in ("trivial", "simple", "main"):
        m = report["intent_metrics"][name]
        lines.append(f"| {name} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} |")
    lines.append("")

    lines.append("## Escalation decision\n")
    lines.append("| System | Precision | Recall | F1 | Cost-weighted |")
    lines.append("|---|---|---|---|---|")
    for name in ("trivial", "simple", "main"):
        m = report["escalation_metrics"][name]
        lines.append(
            f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['cost_weighted_score']:.3f} |"
        )
    lines.append("")

    if report["judge_summary"]:
        lines.append("## LLM judge (reply quality, 1-5)\n")
        for k, v in report["judge_summary"].items():
            lines.append(f"- {k}: {v:.2f}")
        lines.append("")

    if report["human_agreement"]:
        lines.append("## Human-agreement study\n")
        ha = report["human_agreement"]
        lines.append(f"- n = {ha['n']}")
        lines.append(f"- Cohen's kappa: {ha['cohen_kappa']:.3f}")
        lines.append(f"- Spearman r: {ha['spearman_r']:.3f} (p={ha['spearman_p']:.4f})")
        lines.append(f"- Mean absolute diff: {ha['mean_absolute_diff']:.3f}")
        lines.append("")
    else:
        lines.append("## Human-agreement study\n")
        lines.append(
            "Not run — no human judge scores found at "
            f"`{HUMAN_JUDGE_SCORES_PATH.relative_to(ARTIFACTS_DIR.parent)}`. "
            "TRD 7.3 marks this mandatory before trusting the judge for headline numbers.\n"
        )

    lines.append("## What's misleading about my headline number\n")
    lines.append(report["misleading_number_note"] + "\n")

    if report["failure_examples"]:
        lines.append("## Top failure examples\n")
        for f in report["failure_examples"]:
            lines.append(f"- **{f['thread_id']}**: \"{f['customer_msg']}\"")
            lines.append(
                f"  - intent: true=`{f['true_intent']}` pred=`{f['pred_intent']}`, "
                f"escalation: true=`{f['true_escalation']}` pred=`{f['pred_escalation']}`"
            )
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    report = run()
    print(f"wrote {REPORT_JSON} and {REPORT_MD}")
    print(f"golden set size: {report['golden_set_size']}")
