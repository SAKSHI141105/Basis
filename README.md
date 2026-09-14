# AI Support Agent — AppleSupport

An AI support copilot for one brand (`AppleSupport`) from the [Customer Support on
Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset: it classifies incoming customer messages into a data-derived intent
taxonomy, drafts a reply grounded in real historically-resolved precedent, and
decides auto-handle vs. escalate-to-human with a stated reason.

See [`PRD.md`](PRD.md) for the product spec, [`TRD.md`](TRD.md) for technical
decisions, [`ARCHITECTURE.md`](ARCHITECTURE.md) for the system diagram, and
[`DECISION_LOG.md`](DECISION_LOG.md) for real decisions made (and constraints
discovered) while actually building this against the live API and real data.

> **Status:** under active build, following the milestone order in `PRD.md`
> §9. This README is updated in the same commit as any setup/run step change
> (see `AGENT_WORKING_AGREEMENT.md` §4) — if a step below doesn't work, that's
> a bug, not stale docs.

## Quickstart

```bash
git clone <this-repo>
cd Basis
make setup
```

1. **Get the dataset.** Download `twcs.csv` from
   [Kaggle](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
   and place it at `data/raw/twcs.csv`. (Kaggle requires a free account; if you
   don't want to configure the Kaggle API, the "Download" button on the
   dataset page works too — no CLI needed.)
2. **Set up the environment.**
   ```bash
   cp .env.example .env
   # fill in GEMINI_API_KEY — https://aistudio.google.com/apikey (free, no card)
   ```
3. **Run the offline pipeline** (ingest → clean → cluster → label with the
   frozen taxonomy → split → build index → train baselines):
   ```bash
   make pipeline
   ```
   `taxonomy.yaml` itself is already committed/frozen from a real clustering
   run over the full dataset (see its header comment and `DECISION_LOG.md`
   for how the 8 intents + out_of_scope were derived) — `make pipeline`
   re-derives clusters from your local run and maps them back onto that
   frozen taxonomy via `pipeline/apply_taxonomy.py`.
4. **Run the eval harness** against the committed golden set
   (`golden_set.jsonl`):
   ```bash
   make eval-fast   # replays the committed LLM-response cache, no API calls
   make eval-live   # re-calls the real Gemini API for real, ~30-60 min
   ```
   This produces `artifacts/eval_report.json` and `eval_report.md`.

## Two run modes, honestly labeled

| Mode | Command | What it does |
|---|---|---|
| Fast (default) | `make eval-fast` | Replays the committed LLM-response cache for the golden set — no network calls |
| Live | `make eval-live` | Re-calls the real Gemini API, with rate-limit pacing and retry on transient errors |

No `GEMINI_API_KEY` set → the pipeline falls back to a local Ollama model for
generation/judging (slower; see `.env.example`). This fallback is not held to
the same reproducibility bar as the Gemini path.

## A real constraint discovered building this, not assumed upfront

`gemini-3.5-flash` (the model TRD's tiering originally intended for
generation/judging) carries only a **20 requests/day** free-tier quota on a
freshly created API key — nowhere near enough for a 198-example golden-set
run, and not documented anywhere discoverable without an authenticated AI
Studio session. Classification, generation, *and* judging all run on
`gemini-3.5-flash-lite` instead, which doesn't hit this ceiling. This is a
real, disclosed trade-off (weaker generation/judge quality than the
originally-intended split) — see `DECISION_LOG.md` for the full account.

## Golden set: how it was actually labeled

`golden_set.jsonl` (198 examples, stratified by intent with deliberate
oversampling of hard/escalation-like cases — see `pipeline/golden_set.py`)
is **mostly AI-labeled, not independently hand-labeled by a human** — 6
examples carry real human labels (`label_source: "human"`), the remaining
192 were labeled by Gemini applying the same rubric a human would
(`labeling_guide.md`), tagged `label_source: "ai_generated"`.

This is disclosed prominently, not hidden, because it's a real limitation:
using an LLM to generate ground truth that then grades an LLM-based
classifier and LLM judge risks correlated errors — the two could agree with
each other's mistakes in a way a human never would. See `DECISION_LOG.md`
and the eval report's "what's misleading about my headline number" section.

## Project layout

```
pipeline/   offline pipeline (ingest, clean, taxonomy, index, baselines, llm_client)
service/    FastAPI serving layer (/classify, /draft-reply, /decide, /pipeline)
eval/       evaluation harness (golden set, metrics, LLM judge, human-agreement study)
web/        Next.js demo frontend (not yet built — see AGENT_WORKING_AGREEMENT.md §1 milestone order)
tests/      pytest unit + integration tests (no API key needed — fully mocked)
```

## Known limitations (stated up front, not buried)

- **Golden-set labels are mostly AI-generated**, not human-labeled — see
  above. This is the single biggest caveat on every headline eval number.
- The **human-agreement study** required by `TRD.md` §7.3 (comparing the LLM
  judge's reply-quality scores against an independent human rater) could not
  be performed for the same reason — using another LLM as a stand-in for
  "the human" would be circular, not weaker evidence, so it's marked
  explicitly not-performed in the eval report rather than faked.
- The "resolved" heuristic (silence or a closure phrase after the brand's
  reply) is weak — silence is not proof of satisfaction. See `TRD.md` §2.3.
- `out_of_scope` is ~88% of real traffic by construction (see
  `taxonomy.yaml`'s header) — this makes raw accuracy a misleading headline
  metric; macro-F1 is what actually reflects classification quality here.
- No PII redaction pipeline. Apple was chosen partly to reduce this need, but
  it's a known gap, not a guarantee.
- No live system integration, no multi-language support, no fine-tuning —
  see `PRD.md` §7 for the full out-of-scope list.

## Development

```bash
make test        # unit + integration tests (no API key needed)
make run-api      # local FastAPI service on :8000
```
