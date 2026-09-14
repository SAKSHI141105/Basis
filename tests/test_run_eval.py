import json

import numpy as np
import pandas as pd

from eval.run_eval import _build_misleading_number_note, render_markdown, run
from pipeline.classify import build_classification_prompt
from pipeline.generate import build_generation_prompt
from pipeline.index import RetrievalIndex
from pipeline.llm_client import DiskCache, LLMClient, _cache_key
from eval.judge import build_judge_prompt
from service.state import AppState

TAXONOMY = [
    {"id": "device_help", "description": "device troubleshooting", "examples": ["phone won't turn on"]},
    {"id": "billing", "description": "billing issues", "examples": ["overcharged"]},
]

GOLDEN_SET = [
    {
        "thread_id": "t1",
        "customer_msg": "my iphone screen is frozen and won't respond",
        "true_intent": "device_help",
        "true_escalation": "auto_handle",
    },
    {
        "thread_id": "t2",
        "customer_msg": "you charged me twice for my subscription",
        "true_intent": "billing",
        "true_escalation": "escalate",
    },
]


class _FakeSimpleModel:
    def predict(self, messages):
        return ["device_help" if "iphone" in m or "screen" in m else "billing" for m in messages]


def _fake_embed_fn(messages):
    vectors = []
    for msg in messages:
        rng = np.random.default_rng(abs(hash(msg)) % (2**32))
        vectors.append(rng.normal(size=4))
    return np.asarray(vectors)


def _build_fake_state(tmp_path):
    embeddings = _fake_embed_fn(["precedent device", "precedent billing"])
    metadata = pd.DataFrame(
        {
            "thread_id": ["p1", "p2"],
            "brand_reply_clean": ["Try restarting your device.", "We'll review your billing statement."],
            "intent": ["device_help", "billing"],
        }
    )
    index = RetrievalIndex(embeddings, metadata)
    cache = DiskCache(tmp_path / "cache.jsonl")
    client = LLMClient(cache=cache, mode="fast")

    classify_responses = {
        GOLDEN_SET[0]["customer_msg"]: {"intent": "device_help", "confidence": 0.9, "all_scores": {"device_help": 0.9, "billing": 0.05}},
        GOLDEN_SET[1]["customer_msg"]: {"intent": "billing", "confidence": 0.4, "all_scores": {"device_help": 0.1, "billing": 0.4}},
    }
    for msg, resp in classify_responses.items():
        prompt = build_classification_prompt(msg, TAXONOMY)
        key = _cache_key("gemini-3.5-flash-lite", prompt, {"temperature": 0.0})
        cache.set(key, "gemini-3.5-flash-lite", prompt, {"temperature": 0.0}, json.dumps(resp))

    style_guide = "Warm but concise, one concrete next step."
    for msg, intent in [(GOLDEN_SET[0]["customer_msg"], "device_help"), (GOLDEN_SET[1]["customer_msg"], "billing")]:
        precedents = index.query(np.asarray(_fake_embed_fn([msg])[0]), k=3, intent=intent)
        gen_prompt = build_generation_prompt(msg, precedents, style_guide)
        gen_key = _cache_key("gemini-3.5-flash", gen_prompt, {"temperature": 0.4})
        cache.set(
            gen_key, "gemini-3.5-flash", gen_prompt, {"temperature": 0.4},
            json.dumps({"draft": "Here's a next step.", "grounded_on": [precedents[0].thread_id]}),
        )
        judge_prompt = build_judge_prompt(msg, "Here's a next step.", precedents)
        judge_key = _cache_key("gemini-3.5-flash", judge_prompt, {"temperature": 0.0})
        cache.set(
            judge_key, "gemini-3.5-flash", judge_prompt, {"temperature": 0.0},
            json.dumps({"groundedness": 4, "correctness": 4, "tone": 4, "actionability": 4, "rationale": "fine"}),
        )

    return AppState(
        taxonomy=TAXONOMY,
        retrieval_index=index,
        embed_fn=_fake_embed_fn,
        llm_client=client,
        brand_style_guide=style_guide,
        classifier_model="gemini-3.5-flash-lite",
        generation_model="gemini-3.5-flash",
    )


def test_run_eval_produces_full_report(tmp_path, monkeypatch):
    monkeypatch.setenv("JUDGE_MODEL", "gemini-3.5-flash")
    state = _build_fake_state(tmp_path)

    report = run(
        golden_set=GOLDEN_SET,
        state=state,
        simple_model=_FakeSimpleModel(),
        trivial_label="device_help",
        human_scores_path=tmp_path / "human_judge_scores.json",
        report_json_path=tmp_path / "eval_report.json",
        report_md_path=tmp_path / "eval_report.md",
    )

    assert (tmp_path / "eval_report.json").exists()
    assert (tmp_path / "eval_report.md").exists()

    assert report["golden_set_size"] == 2
    assert set(report["intent_metrics"].keys()) == {"trivial", "simple", "main"}
    assert set(report["escalation_metrics"].keys()) == {"trivial", "simple", "main"}
    assert len(report["judge_scores"]) == 2
    assert "mean_overall" in report["judge_summary"]
    assert report["human_agreement"] is None  # no human scores file in this test
    assert "misleading" in report["misleading_number_note"].lower()


def test_run_eval_skips_example_on_failure_instead_of_crashing(tmp_path, monkeypatch):
    monkeypatch.setenv("JUDGE_MODEL", "gemini-3.5-flash")
    state = _build_fake_state(tmp_path)

    golden_set_with_uncached = GOLDEN_SET + [
        {
            "thread_id": "t3",
            "customer_msg": "this message was never cached, so its classify call will fail",
            "true_intent": "billing",
            "true_escalation": "auto_handle",
        }
    ]

    report = run(
        golden_set=golden_set_with_uncached,
        state=state,
        simple_model=_FakeSimpleModel(),
        trivial_label="device_help",
        human_scores_path=tmp_path / "human_judge_scores.json",
        report_json_path=tmp_path / "eval_report.json",
        report_md_path=tmp_path / "eval_report.md",
    )

    assert report["golden_set_size"] == 3
    assert report["skipped_example_ids"] == ["t3"]
    assert report["main_system_examples_evaluated"] == 2
    # trivial/simple baselines still scored on all 3; main system only on 2
    assert len(report["judge_scores"]) == 2

    md_text = (tmp_path / "eval_report.md").read_text(encoding="utf-8")
    assert "1 example(s) were skipped" in md_text


def test_misleading_note_flags_falsely_strong_trivial_accuracy():
    # trivial: high accuracy, near-zero macro-F1 -> imbalanced-traffic story
    intent_metrics = {
        "trivial": {"accuracy": 0.88, "macro_f1": 0.10},
        "simple": {"accuracy": 0.80, "macro_f1": 0.47},
        "main": {"accuracy": 0.90, "macro_f1": 0.75},
    }
    note = _build_misleading_number_note(intent_metrics)
    assert "0.880" in note
    assert "useless" in note


def test_misleading_note_flags_balanced_golden_set_story():
    # trivial: low accuracy on a stratified golden set -> opposite story
    intent_metrics = {
        "trivial": {"accuracy": 0.05, "macro_f1": 0.01},
        "simple": {"accuracy": 0.67, "macro_f1": 0.67},
        "main": {"accuracy": 0.84, "macro_f1": 0.69},
    }
    note = _build_misleading_number_note(intent_metrics)
    assert "0.050" in note
    assert "stratified" in note
    assert "misleading" in note.lower()


def test_render_markdown_produces_readable_sections(tmp_path):
    report = {
        "golden_set_size": 2,
        "intent_metrics": {
            "trivial": {"accuracy": 0.5, "macro_f1": 0.3},
            "simple": {"accuracy": 0.6, "macro_f1": 0.4},
            "main": {"accuracy": 0.9, "macro_f1": 0.8},
        },
        "escalation_metrics": {
            "trivial": {"precision": 0.5, "recall": 1.0, "f1": 0.6, "cost_weighted_score": 0.5},
            "simple": {"precision": 0.6, "recall": 0.7, "f1": 0.65, "cost_weighted_score": 0.6},
            "main": {"precision": 0.9, "recall": 0.9, "f1": 0.9, "cost_weighted_score": 0.9},
        },
        "judge_scores": [],
        "judge_summary": {},
        "human_agreement": None,
        "failure_examples": [
            {"thread_id": "t2", "customer_msg": "hi", "true_intent": "billing", "pred_intent": "device_help",
             "true_escalation": "escalate", "pred_escalation": "auto_handle"}
        ],
        "misleading_number_note": "example note",
    }
    out_path = tmp_path / "eval_report.md"
    render_markdown(report, out_path)
    text = out_path.read_text(encoding="utf-8")
    assert "Evaluation Report" in text
    assert "0.900" in text
    assert "Human-agreement study" in text
    assert "example note" in text
