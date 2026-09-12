# TRD — AI Support Agent for AppleSupport

## 1. Stack Decisions

| Layer | Choice | Why |
|---|---|---|
| Backend / pipeline | Python 3.11, FastAPI | Every hard part (embeddings, clustering, sklearn baselines, stats) is Python-native; FastAPI gives typed request/response contracts for free via Pydantic |
| Data handling | pandas, pyarrow | Standard, fast enough for ~3M row filter → single-brand subset |
| Embeddings | `sentence-transformers` (e.g. `all-MiniLM-L6-v2` or `bge-small-en-v1.5`) | Local, free, fast enough to embed the full brand subset without API cost |
| Clustering (taxonomy derivation) | HDBSCAN (fallback: KMeans with silhouette-selected k, auto-triggered if HDBSCAN's compiled wheel fails to install on the reviewer's machine) | HDBSCAN doesn't force a cluster count and naturally produces a noise/"other" bucket — matches the requirement for an out-of-scope intent |
| Vector index (retrieval) | **Flat, in-memory cosine-similarity index** — normalized embeddings as a single `.npy` matrix + a parallel metadata parquet, similarity via matrix multiply (or `sklearn.neighbors.NearestNeighbors`) | See §1.1 below — a dedicated vector DB is the wrong tool at this data scale and introduces avoidable risk |
| Simple baseline classifier | scikit-learn: TF-IDF + Logistic Regression | Cheap, fast, genuinely "simple" — a real baseline, not a strawman |
| LLM (classification, generation, judge) | **Google Gemini API** (free tier) — `gemini-2.5-flash-lite` for classification, `gemini-2.5-flash` for reply generation and the judge; documented local fallback (e.g. via Ollama) if no key is set | See §10 — model split and free-tier limits are load-bearing design decisions, not an afterthought |

### 1.1 Why not ChromaDB (or any vector DB) for this project

This was reconsidered deliberately, not defaulted into. At this project's scale (one brand's resolved threads — realistically tens of thousands of rows, not millions), a vector database is solving a problem this project doesn't have, and it introduces three concrete risks to the things this project *does* need to guarantee:

- **Reproducibility risk:** ChromaDB pulls in a heavy dependency tree (onnxruntime, hnswlib, sqlite/duckdb) with a history of platform-specific wheel issues (Apple Silicon, Windows) — exactly the kind of thing that can silently break the "clone → run in 15 minutes" promise on a reviewer's machine you don't control.
- **Hidden network dependency:** Chroma phones home via telemetry by default (`ANONYMIZED_TELEMETRY`) unless explicitly disabled — a surprising, easy-to-miss network call in a project that's supposed to run offline-reproducibly and for $0.
- **Unneeded abstraction:** the index is **frozen after the offline pipeline runs** — it is never written to during serving. A client-server vector DB is built for concurrent read/write at scale; this project needs a static, read-only nearest-neighbor lookup, which a normalized embedding matrix and a matrix multiply does in single-digit milliseconds at this row count, with zero extra dependencies and zero network calls.

**Documented upgrade path (don't build now):** if the brand subset ever grew past the point an in-memory matrix is comfortable (rough rule of thumb: past ~1–2M rows depending on embedding dimension and available RAM), the natural next step is FAISS (`IndexFlatIP` or `IndexIVFFlat`) — same in-process, no-server philosophy, just a faster approximate search. This is noted in the decision log as a considered-and-deferred choice, not a gap.
| Testing | pytest | Unit tests for pipeline stages, not just end-to-end smoke test |
| Env/reproducibility | `uv` or `pip` + `requirements.txt`/`pyproject.toml`, `.env.example`, a single `make run-eval` or `python -m pipeline.run_eval` entrypoint | One command, no notebook required for headline path |
| Frontend (demo, built last) | Next.js (App Router) + Tailwind, calling the FastAPI service | See `DESIGN_BRIEF.md` and `ARCHITECTURE.md` |

## 2. Data Pipeline

### 2.1 Ingestion
- Download `twcs.csv` from Kaggle (`thoughtvector/customer-support-on-twitter`).
- Filter to rows where `author_id == "AppleSupport"` (brand replies) **and** their linked customer messages via `in_response_to_tweet_id` / `response_tweet_id` chains.
- Reconstruct threads: customer message → AppleSupport reply → (optional) further customer follow-up, up to a max depth (e.g. 4 turns) to bound complexity.

### 2.2 Cleaning
- Strip @handles used purely for threading (keep if semantically relevant), expand common shorthand only where it aids embedding quality (don't over-clean — noise is part of the assignment's premise).
- Deduplicate near-identical customer messages (common in this dataset from repeated contact).
- Drop threads with no brand reply (nothing to ground on).
- Persist a `threads.parquet` with columns: `thread_id, customer_msg, customer_msg_clean, brand_reply, brand_reply_clean, turn_count, has_followup, timestamp`.

### 2.3 Resolution Heuristic (and its documented weakness)
- Heuristic "resolved" = brand replied **and** no further customer message in that thread within N hours, **or** customer's last message contains a closure signal (e.g. "thanks", "got it", "resolved").
- This heuristic is weak (silence ≠ satisfaction) — **must be stated as a limitation in the report**, not hidden.

## 3. Intent Taxonomy Derivation

1. Embed all `customer_msg_clean` values (encoder above).
2. Cluster with HDBSCAN (`min_cluster_size` tuned so cluster count lands in the 8–12 range before merging near-duplicates).
3. For each cluster: sample 15–20 messages, manually read and name the intent (e.g. `device_troubleshooting`, `software_update_bug`, `apple_id_account_access`, `billing_subscription`, `repair_order_status`, `feature_how_to`, `general_complaint`, `positive_feedback`, `out_of_scope`).
4. Merge clusters that clearly represent the same intent; keep HDBSCAN's noise points as the seed for the `out_of_scope` bucket, topped up with manually-spotted spam/off-topic examples.
5. Freeze the taxonomy into `taxonomy.yaml` (id, name, description, 3 example utterances) — this file is the single source of truth used by baselines, classifier, prompts, and labeling instructions.

## 4. Classification Module

### 4.1 Baselines (both required, both must actually run in the harness — not just described)
- **Trivial baseline:** predicts the majority-class intent for every input. Establishes the floor.
- **Simple baseline:** TF-IDF (1–2 grams) + Logistic Regression, trained on a labeled training split distinct from the golden eval set.

### 4.2 Main Classifier
- Option A (default, simpler to justify): few-shot prompted LLM classifier using `taxonomy.yaml`'s descriptions/examples in-context, temperature ≈0, forced to output one of the fixed labels + confidence.
- Option B (documented alternative in decision log): embedding similarity to per-intent centroid/nearest-neighbors as a non-LLM classifier — cheaper, useful to compare against Option A.
- Implement Option A as primary; implement Option B if time allows, since comparing an LLM classifier against a pure-embedding classifier is itself a good "two baselines" style story beyond the minimum requirement.

### 4.3 API Contract
```
POST /classify
Request:  { "message": str }
Response: { "intent": str, "confidence": float, "all_scores": {intent: float} }
```

## 5. Retrieval-Grounded Reply Generation

### 5.1 Index
- Build a flat embedding matrix of **resolved** thread pairs (`customer_msg_clean` embedded as rows, L2-normalized), with a parallel metadata table (`thread_id, brand_reply_clean, intent`) — see §1.1 for why this is a matrix, not a vector DB. Scoped to the training split (never include golden-eval threads in the index — that would leak answers).

### 5.2 Retrieval
- On a new message: embed it, filter candidates to the predicted intent (or top-2 intents if confidence is low), compute cosine similarity as a dot product against the (pre-normalized) matrix, take top-k (k=3 default), record the similarity scores (used later by the escalation engine).

### 5.3 Generation
- Prompt template includes: the customer message, the k retrieved precedent pairs (verbatim), and a short brand style guide (derived by summarizing ~20 real AppleSupport replies — tone, length, structure, sign-off pattern).
- Model must **cite which precedent(s) most influenced the draft** in a structured field (not user-facing, used for groundedness scoring).
- Output logged with: draft text, precedent ids used, retrieval similarity scores.

### 5.4 API Contract
```
POST /draft-reply
Request:  { "message": str, "intent": str (optional, else auto-classified) }
Response: { "draft": str, "grounded_on": [thread_id, ...], "retrieval_scores": [float, ...] }
```

## 6. Escalation Decision Engine

### 6.1 Signals (all computed, all logged — not just the final label)
- `intent_confidence` (from classifier)
- `max_retrieval_similarity` (from retrieval step — low value = "we have no good precedent")
- `risk_flags`: regex/keyword hits for safety, legal, data-loss, self-harm-adjacent, explicit "talk to a human" requests
- `sentiment_delta`: simple sentiment score on the message vs. prior turns in the thread, flags escalating frustration
- `contact_count`: number of prior customer messages in this thread (repeated contact signal)

### 6.2 Decision Logic
- Deterministic rule layer first (hard triggers: risk_flags present, explicit human request → always escalate).
- Otherwise, a threshold combination over `intent_confidence` and `max_retrieval_similarity` (thresholds tuned against the golden set, documented, not hardcoded blindly).
- Reason string generation: templated, not free-form LLM prose, so reasons stay auditable (e.g., `"escalated: retrieval_similarity=0.31 below threshold 0.55; no strong precedent found"`).

### 6.3 API Contract
```
POST /decide
Request:  { "message": str, "thread_context": [str, ...] (optional) }
Response: { "decision": "auto_handle" | "escalate", "reason": str, "signals": {...} }
```

## 7. Evaluation Harness

### 7.1 Golden Set
- 150–250 examples sampled **stratified by derived intent**, with deliberate oversampling of: low-confidence-looking messages, multi-turn threads, and messages resembling escalation triggers — so the set isn't dominated by easy cases.
- Each example hand-labeled with: true intent, true escalation decision (author's own judgment call, documented), and a short note when the "right" answer is ambiguous.
- Store as `golden_set.jsonl`, one labeling pass by the author, with a documented rubric (`labeling_guide.md`) so the method is reproducible even if the label itself is subjective.

### 7.2 Automated Metrics
- Intent: accuracy, macro-F1, confusion matrix, broken out per-intent (not just one aggregate number).
- Escalation: precision/recall/F1 treating `escalate` as positive class, plus a cost-weighted variant reflecting §6 of the PRD (false auto-handle weighted worse than false escalate).
- All metrics computed for: trivial baseline, simple baseline, and main system — same golden set, same script, one table.

### 7.3 LLM-as-Judge for Reply Quality
- Rubric dimensions (1–5 each): **groundedness** (does the draft actually reflect the retrieved precedent, or hallucinate beyond it), **correctness/helpfulness**, **tone/brand-fit**, **actionability**.
- Judge prompt includes the retrieved precedents so groundedness is checkable, not vibes-based.
- **Human-agreement study (mandatory):** author manually scores the same rubric on a random 40–50 example subsample; report inter-rater agreement (Cohen's kappa for categorical bucketing of scores, or Spearman correlation for raw scores) between judge and human. If agreement is weak, say so — this is exactly the kind of honesty the assignment is scoring for.

### 7.4 Entry point
- `python -m pipeline.run_eval` (or `make eval`) runs: baselines → main system → metrics table → judge scoring → human-agreement report, writing `eval_report.json` + a rendered `eval_report.md`, in under 15 minutes wall-clock on the golden set.

## 8. LLM Usage Strategy — Free Tier Is a Real Constraint, Design For It Now

This is the single biggest hidden bottleneck in the whole project if not planned upfront, so it's called out on its own rather than buried in a config table.

### 8.1 Model choice and why
Gemini's free tier is genuinely usable long-term (no credit card, no 30-day expiry) but the different model tiers trade capability for throughput very differently:

| Model | Free-tier RPM (approx.) | Free-tier RPD (approx.) | Use for |
|---|---|---|---|
| `gemini-2.5-flash-lite` | ~15–30 | ~1,000–1,500 | **Intent classification** — highest volume of calls, task is simple enough not to need more |
| `gemini-2.5-flash` | ~10 | ~500–1,500 | **Reply generation + LLM judge** — needs more reasoning quality; fewer calls needed |
| `gemini-2.5-pro` | ~5 | ~25–100 | **Do not use in the core pipeline or eval harness** — the RPD ceiling alone (as low as 25/day on some tiers) makes it incompatible with even one full golden-set run. Optional: a handful of manual spot-checks only, never part of an automated script |

(Exact numbers fluctuate — Google has changed free-tier quotas multiple times; the agent should read the current limits from Google's official rate-limit page at implementation time and adjust the throttling constants below accordingly, rather than hardcoding these.)

### 8.2 The actual bottleneck: eval harness call volume vs. RPM/RPD
A single golden-set run of 200 examples needs roughly: 200 classification calls + 200 generation calls + 200×(1 or more) judge calls ≈ 600–800+ LLM calls. At ~10–15 RPM that is **40–80 minutes of pure rate-limit waiting**, which directly conflicts with the PRD's "under 15 minutes" reproducibility promise if not designed around. Do not discover this at the end of the project — build the fix in from Day 1:

1. **Disk-backed response cache, built first, used everywhere.** Every LLM call (classification, generation, judge) is wrapped in a cache keyed by `hash(model + prompt + params)`, persisted to a small local file (`.cache/llm_cache.jsonl` or sqlite). This does two things: (a) reruns of the eval harness are instant and free after the first run, and (b) it makes the eval harness deterministic-enough for grading even though the underlying model calls are not.
2. **Two documented run modes in the README:**
   - **`make eval-fast`** (default, and the one used for the 15-minute reproducibility claim): runs against a **committed cache** of the golden-set LLM responses shipped in the repo (small JSON file, a few hundred KB) — reviewer sees full results instantly, with the cache file clearly labeled as "precomputed golden-set responses, see §8.3 of TRD for how they were generated."
   - **`make eval-live`**: re-calls the real API respecting rate limits (with backoff/pacing built in), documented honestly as taking up to ~60–90 minutes on the free tier for a full golden-set run — this is the "don't just trust our cache, verify it yourself" path, and its existence (not its speed) is what proves the numbers are real.
3. **Request pacing, not just retry-on-429.** Implement a simple token-bucket / sleep-based pacer tuned to the chosen model's documented RPM, so the live run degrades gracefully instead of hammering the API and burning the RPD budget on failed retries.
4. **Batch classification calls where the API allows it** (e.g., grouping several short messages into one prompt with structured JSON output) to reduce total request count against RPM/RPD — worth doing for classification specifically since it's the highest-volume call type.

### 8.3 Why this belongs in the architecture, not just the eval script
The caching layer must be a shared utility (`llm_client.py`) used by `/classify`, `/draft-reply`, `/decide`, and the eval harness alike — not an eval-only hack — because the live FastAPI service hitting free-tier limits during a demo (e.g., the frontend's live demo panel, §3.2 of the Design Brief) is exactly the kind of thing that looks broken to a reviewer if it wasn't planned for. Building this on Day 1 avoids an end-of-project scramble to retrofit caching/pacing into three already-written endpoints.

## 9. Testing Strategy
- Unit tests: thread reconstruction correctness on a small fixture CSV, cleaning function idempotence, retrieval index returns expected neighbor on a synthetic case, escalation rule triggers fire on constructed inputs (risk keyword → always escalate, regardless of other signals).
- One integration test: full pipeline on a 5-row fixture, asserting shapes/types of every API response, not exact LLM output text (non-deterministic).

## 10. Config & Secrets
- `.env.example` with `GEMINI_API_KEY`, `CLASSIFIER_MODEL=gemini-2.5-flash-lite`, `GENERATION_MODEL=gemini-2.5-flash`, `JUDGE_MODEL=gemini-2.5-flash`, `EMBEDDING_MODEL=all-MiniLM-L6-v2` — never commit a real key.
- Explicit fallback path documented in README: if no `GEMINI_API_KEY` is set, pipeline falls back to a local open model (e.g. via Ollama) for generation/judging — slower, and the README should state the adjusted expectation for that path rather than claim the same 15-minute bar.
- Since the whole project must be $0-cost end to end: confirm at implementation time that Google AI Studio's Gemini free tier is available without a billing account attached in the account/region used (billing-linked accounts get *higher* limits per §8.1, but the point of this project is to prove it works on the free, no-card tier).
