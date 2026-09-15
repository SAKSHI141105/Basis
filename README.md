# AI Support Agent — AppleSupport

An AI support copilot for one brand (`AppleSupport`) from the [Customer Support on
Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset: it classifies incoming customer messages into a data-derived intent
taxonomy, drafts a reply grounded in real historically-resolved precedent, and
decides auto-handle vs. escalate-to-human with a stated reason.

**Start with [`REPORT.md`](REPORT.md)** — the full account of methodology,
final results, and (most importantly) where this system is weaker than its
headline numbers suggest. This README covers setup and reproduction only.

See [`PRD.md`](PRD.md) for the product spec, [`TRD.md`](TRD.md) for technical
decisions, [`ARCHITECTURE.md`](ARCHITECTURE.md) for the system diagram, and
[`DECISION_LOG.md`](DECISION_LOG.md) for the chronological account of real
decisions made (and constraints discovered) while actually building this
against the live API and real data.

> **Status:** core deliverables complete — the full golden-set eval has run
> to completion (198/198 examples, 198/198 judge scores), reproducible
> instantly via `make eval-fast`. Deployment (`AGENT_WORKING_AGREEMENT.md`
> §5) is the one remaining optional step. This README is updated in the same
> commit as any setup/run step change (see `AGENT_WORKING_AGREEMENT.md` §4)
> — if a step below doesn't work, that's a bug, not stale docs.

## The headline number, and why it isn't the whole story

Main intent classifier: **83.3% accuracy, 0.830 macro-F1** (vs. 0.011
macro-F1 for a trivial baseline, 0.670 for TF-IDF+LogReg). Escalation
recall, after retuning thresholds against real data: **0.573** (up from an
uncalibrated 0.146).

**Read these next to `REPORT.md` §4, not instead of it:** a spot-check found
the AI-generated labels making up 97% of the golden set agree with real
human labels on intent **0 times out of 6** on direct comparison. Every
number above is real and reproducible, and also measured against ground
truth whose own reliability is genuinely in question — this is the report's
leading disclosed limitation, not a footnote.

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
   make eval-fast   # replays the committed LLM-response cache -- instant, no API calls
   make eval-live   # re-calls the real Gemini API for real
   ```
   This produces `artifacts/eval_report.json` and `eval_report.md`,
   matching the numbers already committed and reported in `REPORT.md`.
5. **Run the demo frontend** (optional — the API + eval report are the core
   deliverables, this is the polish layer per `AGENT_WORKING_AGREEMENT.md` §1):
   ```bash
   make run-api                         # terminal 1 — FastAPI on :8000
   cd web && cp .env.local.example .env.local && npm install && npm run dev
   ```
   Open `http://localhost:3000` — the overview page reads the eval report,
   the live demo panel calls `/pipeline` against a real message from
   `golden_set.jsonl`, and `/eval` reads the same `eval_report.json` as the
   written report (never a live recompute, so the two never disagree).

## Two run modes, honestly labeled

| Mode | Command | What it does |
|---|---|---|
| Fast (default) | `make eval-fast` | Replays the committed LLM-response cache (808 real responses) — instant, no network calls |
| Live | `make eval-live` | Re-calls the real Gemini API, with rate-limit pacing and retry on transient errors |

No `GEMINI_API_KEY` set → the pipeline falls back to a local Ollama model for
generation/judging (slower; see `.env.example`). This fallback is not held to
the same reproducibility bar as the Gemini path.

**Updating the committed fast-mode cache:** after a real `make eval-live`
run (e.g. because you changed a prompt, a threshold, or want to re-verify
against fresh API calls), run `make freeze-fast-cache` to snapshot the
live cache into `.cache/llm_cache_golden.jsonl` — the small, committed
file `make eval-fast` replays. Commit that file alongside the regenerated
`artifacts/eval_report.json`/`eval_report.md` so all three stay in sync.

## Real constraints discovered building this, not assumed upfront

- `gemini-3.5-flash` (the model TRD's tiering originally intended for
  generation/judging) carries only a **20 requests/day** free-tier quota on
  a freshly created key — unusable at any real volume. Everything runs on
  `gemini-3.5-flash-lite` instead.
- `gemini-3.5-flash-lite` itself caps at 500 requests/day, and that quota
  did not behave like a clean daily reset — see `DECISION_LOG.md`'s
  "trickle-refill" entry. The full run was ultimately completed using a
  second, genuinely separate free-tier API key (a new key under the *same*
  Google account shares the same exhausted quota — it has to come from a
  different account).

Full chronological account, including several other fixes discovered the
same way (a PCA-before-clustering fix, an offline-model-loading fix, a
retry-on-transient-error fix), in `DECISION_LOG.md`.

## Golden set: how it was actually labeled

`golden_set.jsonl` (198 examples, stratified by intent with deliberate
oversampling of hard/escalation-like cases — see `pipeline/golden_set.py`)
is **mostly AI-labeled, not independently hand-labeled by a human** — 6
examples carry real human labels (`label_source: "human"`), the remaining
192 were labeled by Gemini applying the same rubric a human would
(`labeling_guide.md`), tagged `label_source: "ai_generated"`.

**This is not just a theoretical risk — it was checked and confirmed.** A
spot-check pointed the AI labeler at the same 6 messages the human labeled
(`artifacts/label_agreement_spotcheck.json`): **0/6 intent agreement**, 3/6
(coin-flip) escalation agreement. This is the single biggest reason to
treat this project's intent-classification headline numbers as provisional.
See `REPORT.md` §4 and §9 for the full discussion.

## Project layout

```
pipeline/   offline pipeline (ingest, clean, taxonomy, index, baselines, llm_client, escalation tuning)
service/    FastAPI serving layer (/classify, /draft-reply, /decide, /pipeline, /eval-report, /samples)
eval/       evaluation harness (golden set, metrics, LLM judge, human-agreement study)
web/        Next.js demo frontend — overview, live demo panel, eval dashboard
tests/      pytest unit + integration tests (no API key needed — fully mocked)
```

## Known limitations (stated up front, not buried)

- **The AI-generated golden-set labels (97% of the set) showed 0/6 intent
  agreement with real human labels on direct spot-check** — see above and
  `REPORT.md` §4. Treat every intent-classification number as provisional.
- The **human-agreement study** required by `TRD.md` §7.3 (comparing the LLM
  judge's reply-quality scores against an independent human rater) could not
  be performed for the same reason — using another LLM as a stand-in for
  "the human" would be circular, not weaker evidence. The LLM judge's very
  high, suspiciously uniform scores (4.88/5 overall) are correspondingly
  unverified.
- Escalation thresholds were originally uncalibrated (recall 0.146) and have
  since been retuned against the golden set (`pipeline/tune_escalation_thresholds.py`,
  now recall 0.573) — a real, verified improvement that still inherits the
  golden-set-labeling caveat above.
- The "resolved" heuristic (silence or a closure phrase after the brand's
  reply) is weak — silence is not proof of satisfaction. Quantified on the
  golden set (`resolution_quality_note` in `golden_set.jsonl`): only 3% of
  "resolved" examples have an actual closure phrase; 79% are silence alone.
  See `TRD.md` §2.3 and `REPORT.md` §2.1.
- `out_of_scope` is ~88% of real traffic by construction (see
  `taxonomy.yaml`'s header) — this makes raw accuracy a misleading headline
  metric on the *natural* distribution; macro-F1 is what actually reflects
  classification quality. (The golden set itself is deliberately balanced,
  not skewed this way — see `REPORT.md` §7 for why both directions matter.)
- No PII redaction pipeline. Apple was chosen partly to reduce this need, but
  it's a known gap, not a guarantee.
- No live system integration, no multi-language support, no fine-tuning —
  see `PRD.md` §7 for the full out-of-scope list.

## Development

```bash
make test        # unit + integration tests (no API key needed)
make run-api      # local FastAPI service on :8000
```
