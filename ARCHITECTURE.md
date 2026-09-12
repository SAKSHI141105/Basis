# Architecture — AI Support Agent for AppleSupport

## 1. System Overview

The system has three clearly separated layers. This separation is itself a decision worth stating in the decision log: **offline pipeline**, **online serving**, and **evaluation harness** are different concerns with different reproducibility requirements, and conflating them (e.g., re-deriving the taxonomy on every request) would break the 15-minute reproducibility bar.

```
                         ┌────────────────────────────┐
                         │        OFFLINE PIPELINE      │
                         │  (run once, artifacts cached) │
                         │                                │
  twcs.csv (Kaggle) ───▶ │ 1. ingest + thread reconstruct │
                         │ 2. clean                       │
                         │ 3. embed + cluster → taxonomy  │
                         │ 4. train split / eval split     │
                         │ 5. build retrieval index         │
                         │    (flat embedding matrix)        │
                         │ 6. train simple baseline (TF-IDF+LR)│
                         └───────────────┬────────────────┘
                                         │ artifacts:
                                         │ taxonomy.yaml, embeddings.npy +
                                         │ index_metadata.parquet,
                                         │ baseline_model.pkl,
                                         │ threads.parquet, llm_cache/
                                         ▼
                         ┌────────────────────────────┐
                         │        ONLINE SERVING        │
                         │        FastAPI service        │
                         │                                │
                         │  POST /classify                │
                         │  POST /draft-reply             │
                         │  POST /decide                  │
                         │  POST /pipeline  (all 3 chained)│
                         └───────────────┬────────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    ▼                                          ▼
        ┌───────────────────────┐                 ┌────────────────────────┐
        │   EVALUATION HARNESS    │                 │     DEMO FRONTEND        │
        │  (CLI, calls the same    │                 │     (Next.js)             │
        │   service functions       │                 │  - live single-message     │
        │   directly, not over HTTP,│                 │    demo panel              │
        │   for speed)               │                 │  - eval dashboard view      │
        │                             │                 │    (reads eval_report.json)│
        │  golden_set.jsonl ─▶ run ─▶ │                 └────────────────────────┘
        │  eval_report.json/.md       │
        └───────────────────────┘
```

## 2. Component Breakdown

### 2.1 Offline Pipeline (`/pipeline`)
- Pure Python modules, each independently testable: `ingest.py`, `clean.py`, `taxonomy.py`, `index.py`, `baselines.py`, `llm_client.py` (the shared, cached, rate-limit-aware Gemini wrapper — see TRD §8; built first, not bolted on later).
- Runs via a single script/Makefile target and writes versioned artifacts to `/artifacts` (gitignored, regenerable — the README's "under 15 minutes" claim covers regenerating these from raw data plus running eval, not committing multi-GB files to the repo).
- **Why offline and online are split:** clustering/taxonomy derivation is not something you want happening per-request — it's a design-time decision, frozen into `taxonomy.yaml`, and treated as a versioned config from then on.

### 2.2 Online Serving (`/service`)
- FastAPI app, thin — it loads the frozen artifacts (taxonomy, index, baseline model) at startup and exposes the three core endpoints plus a convenience `/pipeline` that chains classify → retrieve/draft → decide in one call, matching how the demo frontend will actually use it.
- Stateless per request; thread context (for multi-turn signals like `contact_count`) is passed in by the caller, not stored server-side — keeps the take-home scope honest (no real database/session layer needed).

### 2.3 Evaluation Harness (`/eval`)
- Imports the service's core functions directly (not via HTTP) so the full golden-set run is fast and doesn't require the API server to be up — this is what makes the "under 15 minutes, one command" README promise realistic.
- Produces `eval_report.json` (machine-readable, consumed by the demo dashboard) and `eval_report.md` (human-readable, feeds directly into the Report deliverable).

### 2.4 Demo Frontend (`/web`, built last, optional-but-expected given the Design Brief)
- Next.js app with two views:
  1. **Live demo panel** — paste/select a sample customer tweet, calls `POST /pipeline`, renders intent, the retrieved precedent(s) it was grounded on, the drafted reply, and the escalation decision + reason, side by side.
  2. **Eval dashboard** — reads the static `eval_report.json` (no live recompute needed) and renders the baseline comparison table, confusion matrix, and top-5 failure examples — this is the "convince us it's good" surface, and should be treated as seriously as the live demo, per the assignment's framing that proof matters more than the system.
- Talks to the FastAPI service over a documented `NEXT_PUBLIC_API_URL`; no server-side coupling beyond that, so either piece can be demoed/run independently.

## 3. Data Flow (single request, `/pipeline`)

1. Client sends `{ "message": ..., "thread_context": [...] }`.
2. `/classify` → intent + confidence (LLM few-shot or embedding-NN, per TRD §4).
3. `/draft-reply` → retrieval against the frozen embedding matrix (cosine similarity via dot product) scoped to the predicted intent → LLM generation conditioned on retrieved precedents → draft + `grounded_on` ids + similarity scores.
4. `/decide` → deterministic rule layer + threshold logic over signals gathered in steps 2–3 plus thread-context-derived signals → decision + templated reason string.
5. Response assembled and returned as one JSON payload; every intermediate signal (confidences, similarity scores, risk flags) is included, not just final labels — this is what lets the frontend and the eval harness show *why*, not just *what*.

## 4. Storage Choices

| Data | Store | Why |
|---|---|---|
| Reconstructed threads | Parquet file | Fast columnar read, no DB server needed for a take-home |
| Retrieval index | Flat `.npy` embedding matrix + parquet metadata, loaded into memory at service startup | Zero external services, zero network calls (unlike a vector DB's telemetry), trivially fast at this row count — see TRD §1.1 for the full reasoning on why a vector DB was considered and rejected |
| LLM response cache | JSONL/sqlite keyed by prompt hash | Makes reruns free and fast, and keeps the free-tier RPM/RPD budget from being the bottleneck on every re-run — see TRD §8 |
| Golden set / labels | JSONL | Human-readable, diffable, easy to hand-review in a PR |
| Eval results | JSON + rendered Markdown | JSON feeds the dashboard, Markdown feeds the Report deliverable directly — one source, two consumers |

## 5. Deployment (take-home scope)

- Local-only is sufficient and expected. Provide a `docker-compose.yml` with two services (`api`, `web`) purely for convenience/reproducibility, not because this needs to scale.
- No auth, no rate limiting, no persistence beyond the artifacts above — explicitly state in the report that these are known non-goals for a take-home, not oversights.

## 6. Known Bottlenecks, Thought Through Up Front

Called out here specifically so none of these force an architecture change mid-project — each has a decision already made, not just a risk flagged.

| Bottleneck | Why it would bite late | Decision made now |
|---|---|---|
| Free-tier LLM rate limits (RPM/RPD) vs. eval harness call volume | A 200-example golden-set run needs 600–800+ LLM calls; at free-tier RPM this can take an hour+, which looks broken against the "15 minutes" promise if discovered on the last day | Model tiering (Flash-Lite for classification, Flash for generation/judge, no Pro in the pipeline) + a cached "fast" eval mode as default + a documented slower "live" mode — see TRD §8 |
| Vector DB choice | Retrofitting away from a client-server DB after the retrieval + escalation logic is already built against it would touch three modules at once | Flat in-memory embedding matrix from day one — see TRD §1.1. No server, no telemetry, no dependency risk, and it's actually faster at this data scale |
| Kaggle dataset access requires authentication (`kaggle.json` API token) | First-time Kaggle API setup is a common place reviewers get stuck before ever reaching the actual pipeline | README's setup section handles this explicitly as step 1, with a manual-download fallback link in case the reviewer doesn't want to configure Kaggle API credentials |
| `sentence-transformers` model weights download from Hugging Face on first run | An uncached model download during a "15 minute" reproducibility run is a hidden, un-budgeted delay, and fails entirely with no internet | Pin the exact model name, document expected one-time download size/time in the README, and note that all *subsequent* runs are local-only after that |
| HDBSCAN's compiled wheel occasionally fails to install on some OS/architecture combos | Discovering this on the reviewer's machine, not yours, is the worst time to discover it | Code path tries HDBSCAN first, falls back automatically to KMeans (silhouette-selected k) if the import fails, logged clearly so it's not a silent behavior change |
| Golden-eval threads leaking into the retrieval index or baseline training data | Would invisibly inflate every headline number — exactly the kind of thing the "what's misleading about my number" section is supposed to catch, so it can't be present in the first place | Train/eval split frozen once, immediately after cleaning, before the index or any model is built — enforced by having `index.py` and `baselines.py` only ever read from the training split file, never the full dataset |
| Free hosting cold starts (if doing live deployment, see the Agent Working Agreement doc) | A 30–50 second cold-start delay on the first request looks like a bug during a live demo if not explained | Documented as expected free-tier behavior in both the README and the deployed frontend (a small "waking up the backend, first request may take ~30s" note), not treated as something to engineer around |

## 7. Scalability Notes (brief — not the point of this assignment, but worth one paragraph in the report)

- The retrieval index and baseline model are brand-scoped by construction; multi-brand support would mean namespacing the Chroma collection and taxonomy per brand, not a redesign.
- The LLM calls are the actual bottleneck/cost driver at scale — an embedding-only classifier (Option B in TRD §4.3) would be the first thing to promote to primary if this had to run at real support-team volume, trading a small accuracy hit for large latency/cost wins. Worth one sentence in "what I'd do next with one more week."
