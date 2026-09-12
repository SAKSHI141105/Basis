# AI Support Agent — AppleSupport

An AI support copilot for one brand (`AppleSupport`) from the [Customer Support on
Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset: it classifies incoming customer messages into a data-derived intent
taxonomy, drafts a reply grounded in real historically-resolved precedent, and
decides auto-handle vs. escalate-to-human with a stated reason.

See [`PRD.md`](PRD.md) for the product spec, [`TRD.md`](TRD.md) for technical
decisions, [`ARCHITECTURE.md`](ARCHITECTURE.md) for the system diagram, and
[`DESIGN_BRIEF.md`](DESIGN_BRIEF.md) for the demo frontend's visual direction.

> **Status:** under active build, following the milestone order in `PRD.md`
> §9. This README is updated in the same commit as any setup/run step change
> (see `AGENT_WORKING_AGREEMENT.md` §4) — if a step below doesn't work, that's
> a bug, not stale docs.

## Quickstart (target: clone → headline results in under 15 minutes)

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
3. **Run the fast path** (uses the committed, precomputed golden-set LLM
   response cache — see `TRD.md` §8.2):
   ```bash
   make ingest
   make taxonomy
   make index
   make baselines
   make eval-fast
   ```
   This produces `artifacts/eval_report.json` and `artifacts/eval_report.md`.
4. **Verify it yourself (optional, slower):** `make eval-live` re-calls the
   real Gemini API respecting free-tier rate limits — documented as taking up
   to ~60–90 minutes for a full golden-set run. Its existence, not its speed,
   is what proves the cached numbers are real.

## Two run modes, honestly labeled

| Mode | Command | What it does | Time |
|---|---|---|---|
| Fast (default) | `make eval-fast` | Replays committed, precomputed LLM responses for the golden set | < 15 min |
| Live | `make eval-live` | Re-calls the Gemini free-tier API, with rate-limit pacing | ~60–90 min |

No `GEMINI_API_KEY` set → the pipeline falls back to a local Ollama model for
generation/judging (slower; see `.env.example`). This fallback is not held to
the same 15-minute bar.

## Project layout

```
pipeline/   offline pipeline (ingest, clean, taxonomy, index, baselines, llm_client)
service/    FastAPI serving layer (/classify, /draft-reply, /decide, /pipeline)
eval/       evaluation harness (golden set, metrics, LLM judge, human-agreement study)
web/        Next.js demo frontend (built last, per AGENT_WORKING_AGREEMENT.md)
tests/      pytest unit + integration tests
```

## Known limitations (stated up front, not buried)

- The "resolved" heuristic (silence or a closure phrase after the brand's
  reply) is weak — silence is not proof of satisfaction. See `TRD.md` §2.3.
- No PII redaction pipeline. Apple was chosen partly to reduce this need, but
  it's a known gap, not a guarantee.
- No live system integration, no multi-language support, no fine-tuning —
  see `PRD.md` §7 for the full out-of-scope list.

## Development

```bash
make test        # unit + integration tests
make run-api      # local FastAPI service on :8000
```
