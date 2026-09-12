# PRD — AI Support Agent for AppleSupport

## 1. Problem Statement

Build an AI support agent for **one brand** from the Customer Support on Twitter dataset that:
1. Classifies incoming customer messages into a small, data-derived intent taxonomy.
2. Drafts a reply grounded in how that brand has *historically* resolved similar issues (not a generic LLM reply).
3. Decides auto-handle vs. escalate-to-human, with a stated reason.

The grading bar is not "does it run" — it's **"can you prove it's good enough to trust, and can you tell us honestly where it isn't."** Every deliverable should be built with that audience (a skeptical reviewer) in mind, not with the goal of a high headline number.

## 2. Chosen Brand: `AppleSupport`

**Why this brand over alternatives (Amazon, Uber, Delta, Spotify, etc.):**
- High tweet volume → enough resolved threads to mine precedent from and to sample a 150–250 example golden set without running dry on any intent.
- Issues are largely **self-contained troubleshooting/account/billing** conversations that don't depend on external systems (order numbers, flight PNRs, live GPS state) — so a reply can actually be "resolved" using text alone, which matters because our grounding and judge both operate on text.
- Lower PII density than Amazon/Delta (fewer addresses, card numbers, order IDs), which simplifies both data handling and reply generation without needing a redaction layer as a hard blocker.
- Genuinely messy: overlaps a lot with the assignment's "noisy and imperfect" framing (typos, sarcasm, multi-turn context, partial resolutions, customers cross-posting to other Apple handles).

**Decision log entry (for later):** brand choice traded off "richest taxonomy" (Amazon) against "cleanest ground truth for resolution" (Apple) — we chose resolution cleanliness because the grading bar is about proving quality, not about taxonomy complexity.

## 3. Who This Is For

- **Fictional end user:** a Tier-1 support agent at Apple who would use auto-drafted replies and escalation flags as a copilot, not a fully autonomous system. This framing matters — it shapes what "good enough" means (see §6).

## 4. Functional Requirements

### 4.1 Intent Classification
- Derive **8–10 intents** from the data itself (not invented top-down), via embedding + clustering + manual cluster naming, then validated against a held-out sample.
- Must include an explicit **out-of-scope/other** bucket — real traffic always has some.
- Output: intent label + confidence score.

### 4.2 Grounded Reply Drafting
- For each incoming message, retrieve the **k most similar historically-resolved** customer→brand-reply pairs (same intent, similar embedding).
- Draft a reply conditioned on those precedents and the brand's actual tone/style (derived from real brand replies, not assumed).
- The draft must be **traceable**: log which precedent(s) it was grounded in, so groundedness can be scored later, not just claimed.

### 4.3 Escalation Decision
- Binary decision: `auto_handle` vs `escalate`.
- Must return a **human-readable reason string**, not just a label (e.g., "low retrieval similarity to any precedent (0.31 < 0.55 threshold) + negative sentiment spike after 2nd contact").
- Escalation triggers to implement explicitly: low classifier confidence, low retrieval similarity (no good precedent to ground on), presence of safety/legal/financial-harm language, explicit human request, repeated contact with unresolved sentiment.

### 4.4 Evaluation (this is the actual product)
- Golden set: 150–250 hand-labeled examples (intent + escalation ground truth + reference resolution quality notes), with documented sampling/labeling method.
- Automated metrics harness (accuracy/F1 for intent, precision/recall for escalation) run against **two baselines**: a trivial one (majority class / always-escalate) and a simple one (TF-IDF+LogReg / keyword rules).
- LLM-as-judge rubric for reply quality, with a **human-agreement study** on a subsample proving the judge isn't just agreeing with itself.
- A report section titled **"What is misleading about my headline number"** — mandatory, not optional polish.

## 5. Non-Functional Requirements

- **Reproducibility:** `README.md` must let a reviewer go from clone to headline results in **under 15 minutes** via a default "fast" mode that replays a small, committed cache of real (previously-made) LLM responses — this is the mode the 15-minute claim refers to. A second, honestly-slower "live" mode that re-calls the free-tier API from scratch must also exist and be documented, so the cached numbers are independently verifiable, not just asserted (see `TRD.md` §8.2 and `AGENT_WORKING_AGREEMENT.md` §5). No manual notebook-cell-by-cell execution required for either path (notebooks are fine for exploration, not for the reproducible pipeline).
- **Cost:** the entire project must run at **$0** — Gemini's free tier (no billing account required) for all LLM calls, sentence-transformers for embeddings, no paid hosting. See `TRD.md` §8 for the model-tiering and caching strategy that makes this compatible with the reproducibility bar below despite free-tier rate limits.
- **Determinism:** fixed seeds for clustering/sampling; LLM calls at low temperature for classification/decision, documented temperature choice for generation.

## 6. What "Good" Means For This Brand

Good does **not** mean "the agent never escalates" — for a copilot-framed product, over-escalation is a much cheaper failure than a confidently wrong auto-sent reply. So:
- Escalation **recall** on genuinely hard/risky cases matters more than raw auto-handle rate.
- A wrong auto-handled reply is weighted worse in the report's error analysis than an unnecessary escalation.
- "Good" reply quality = grounded (traceable to a real precedent), on-brand tone, and actionable — not just fluent.

## 7. Explicitly Out of Scope (state this, don't just do it silently)

- No live system integration (no real ticket queue, no sending real tweets).
- No multi-language support — English-only subset of the dataset.
- No fine-tuning of a generation model from scratch — retrieval + prompting only, given the one-week budget.
- No PII redaction pipeline as a hard requirement (brand chosen partly to reduce this need), but flag it as a known gap in the report.
- No online/production-style monitoring — this is an offline-evaluated system.

## 8. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Dataset has no reliable "resolved" signal (last brand reply ≠ proof of resolution) | State this explicitly as a limitation; use heuristics (thread closed, no further customer reply) and disclose their weakness in the report |
| LLM judge agrees with itself, not humans | Run a documented human-agreement study on a subsample (Cohen's kappa / correlation) before trusting the judge for headline numbers |
| Golden set skewed toward easy/common intents | Stratified sampling across derived intents + deliberate oversampling of ambiguous/escalation-worthy cases; disclose sampling method |
| Scope creep into a "full product" instead of a provable prototype | Timebox: pipeline + eval harness first, UI demo last, and only if time remains |

## 9. Milestones (1-week cadence)

1. Day 1: data ingestion, thread reconstruction, brand filter, exploratory intent clustering.
2. Day 2: finalize intent taxonomy, label golden set (start here — labeling always takes longer than expected).
3. Day 3: baselines (trivial + simple) for classification and escalation.
4. Day 4: retrieval index + grounded reply generation + escalation decision engine.
5. Day 5: eval harness (metrics + LLM judge + human-agreement study).
6. Day 6: report, decision log, failure analysis.
7. Day 7: FastAPI wrapper + demo UI (only after everything above is done and reproducible).
