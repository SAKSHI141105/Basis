# Decision Log

Running log of decisions made during the build that deviated from, refined,
or were discovered while implementing `PRD.md`/`TRD.md`/`ARCHITECTURE.md`.
Referenced from those docs where noted; kept here rather than scattered
across commit messages so the reasoning survives in one place.

## Brand choice: AppleSupport

See `PRD.md` §2 — traded off "richest taxonomy" (Amazon) against "cleanest
ground truth for resolution" (Apple); chose resolution cleanliness because
the grading bar is about proving quality, not taxonomy complexity.

## HDBSCAN needs PCA-reduced embeddings, not raw 384-dim vectors

**Discovered:** running real clustering on 103,757 customer messages.
HDBSCAN on raw 384-dim `all-MiniLM-L6-v2` embeddings burned 24+ minutes of
CPU with no result — its tree-based algorithms (boruvka_kdtree etc.) lose
their speedup in high dimensions, forcing a near-brute-force pairwise
distance computation.

**Decision:** reduce to 50 dimensions via PCA before clustering only
(`pipeline/taxonomy.py:_reduce_dims`). The retrieval index is unaffected —
it embeds and indexes at full 384 dimensions separately, since PCA there
would need to be re-fit consistently at query time and buys nothing at this
row count. Clustering time dropped to ~113s.

## Taxonomy: out_of_scope ended up ~88% of traffic

**Discovered:** reading all 30 raw HDBSCAN clusters
(`artifacts/cluster_samples.md`, not committed — regenerable), the noise
bucket alone was ~87% of messages. Several small named clusters turned out
to be the same kind of content: bare "help me" with no context, "check
DM"/"sent a DM", "same issue" with nothing restated, bare iOS version-number
replies, emoji-only replies, and non-English text.

**Decision:** fold those into `out_of_scope` rather than keep them as their
own thin "intents," and disclose the resulting ~88% out_of_scope share in
`taxonomy.yaml`'s header comment rather than retuning HDBSCAN hyperparameters
to chase a nicer-looking distribution. This is real support-traffic
composition, not a clustering bug — most tweets @-ing a brand support handle
are noise, venting, or conversational filler, not fresh actionable requests.
**Consequence, disclosed up front:** this makes the trivial majority-class
baseline look deceptively strong on raw accuracy (see next entry) — exactly
the "misleading headline number" the PRD's report section is about.

## Trivial baseline beats the simple baseline on raw accuracy

**Found running both baselines against the real eval split:**

| | accuracy | macro-F1 |
|---|---|---|
| Trivial (always `out_of_scope`) | 0.881 | 0.104 |
| Simple (TF-IDF + LogReg) | 0.808 | 0.472 |

Because `out_of_scope` is ~88% of the eval set (see previous entry), a
classifier that never predicts anything else scores *higher* raw accuracy
than one that actually tries to distinguish the other 8 intents. Macro-F1
tells the true story. This is kept as the headline example in the eval
report's "what's misleading about my number" section (PRD §4.4) rather than
softened.

## Gemini model names: gemini-2.5-* deprecated for new API keys

**Discovered:** TRD §1/§10 names `gemini-2.5-flash-lite` / `gemini-2.5-flash`
/ `gemini-2.5-pro`. Calling these with a freshly created (2026) API key
returns `404 NOT_FOUND: "This model ... is no longer available to new
users."` The `gemini-3.5-*` series (`gemini-3.5-flash-lite`,
`gemini-3.5-flash`) works and matches the same lite/full tiering intent.

**Decision:** use `gemini-3.5-flash-lite` for classification and
`gemini-3.5-flash` for generation/judge — same reasoning as TRD §8.1 (cheap
high-volume classification vs. higher-quality generation/judging), just
against currently-available model names. `.env.example` and code defaults
updated; `MODEL_RPM` in `pipeline/llm_client.py` keeps conservative
TRD-documented RPM estimates under both old and new names since exact
current free-tier numbers require an authenticated AI Studio dashboard view
this agent can't fetch (TRD §8.1 already anticipates this: "the agent should
read the current limits ... rather than hardcoding these" — done to the
extent verifiable; live-mode pacing stays conservative either way).

## Embedding model must load offline once cached

**Discovered:** `SentenceTransformer(...)` was making live Hugging Face Hub
HEAD requests on every load, even with the model already cached locally —
contradicting the documented promise that only the first run needs network
access (Architecture §6's known-bottlenecks table).

**Decision:** try `local_files_only=True` first, falling back to a normal
(network) load only if nothing is cached yet (`pipeline/taxonomy.py`).

## gemini-3.5-flash's free-tier daily quota is only 20 requests

**Discovered:** running the AI golden-set labeler for real, every call to
`gemini-3.5-flash` started returning `429 RESOURCE_EXHAUSTED` with
`quotaValue: '20'` on the `GenerateRequestsPerDayPerProjectPerModel-FreeTier`
metric — a 20-*requests-per-day* cap, not per-minute. This is far below
what TRD §8.1 assumed for the "flash" tier (500-1,500 RPD historically for
`gemini-2.5-flash`) and isn't published anywhere discoverable without an
authenticated AI Studio view. `gemini-3.5-flash-lite` does not hit this
limit at the same volume.

**Decision:** use `gemini-3.5-flash-lite` for classification, generation,
*and* the judge — collapsing TRD §8.1's lite/full split, because the full
tier's real daily quota makes it unusable for a 200-example golden-set run
regardless of the quality trade-off TRD intended. This is a real, binding
free-tier constraint discovered empirically (exactly the kind of thing
TRD §8.1 flags as needing verification "at implementation time"), not a
judgment call about quality vs. cost. Documented as a limitation: reply
generation and judging both run on the lighter model, so both the
generation quality and the judge's discrimination may be weaker than a
`gemini-3.5-flash`-based pipeline would produce.

**Also discovered:** running two labeling processes concurrently (mine and
the user's, both hitting the same per-project-per-model daily quota)
doubled the burn rate and neither finished before the quota was already
exhausted from earlier live end-to-end testing that same day.

## gemini-3.5-flash-lite ALSO has a hard daily cap (500 requests/day)

**Discovered:** running the real eval harness live (198 examples x 3 calls
each = ~594 needed), it crashed 303 calls in with the same
`RESOURCE_EXHAUSTED` shape, this time `quotaValue: '500'` for
`gemini-3.5-flash-lite`. Combined with the 192 calls already spent labeling
the golden set earlier the same day (plus earlier smoke-test calls), the
day's combined budget for this model was exhausted before judge-scoring
even started (classify+generate finished for ~155 examples, 0 judge calls
made — run_eval.py does all classify+generate first, then all judge calls,
so the crash landed entirely inside the first phase).

**Why this isn't a wasted run:** every successful call is cached to disk by
content hash (`pipeline/llm_client.py`'s `DiskCache`), including across
process restarts. Re-running `make eval-live` after the daily quota resets
replays all ~155 already-done classify/generate calls as free cache hits
and only spends fresh quota on the ~43 remaining classify/generate calls
plus all 198 judge calls (~241 calls) — comfortably within a single day's
500-request budget on a quiet day.

**Decision:** no code change needed here — the caching architecture already
built for exactly this reason (TRD §8.2/8.3) is what makes a free-tier
daily quota survivable across multiple days without losing progress or
re-spending budget. This is disclosed as a real timeline constraint: a full
live golden-set run plus a same-day labeling pass does not fit in one day's
free-tier budget for a freshly created key.

## The quota does not behave like a clean once-daily reset to 500

**Discovered:** on the calendar day after exhausting the quota, a single
manual probe call succeeded ("OK"), but the very next call — made seconds
later, as part of resuming the eval run — immediately failed with the same
`RESOURCE_EXHAUSTED` error. If the quota had cleanly reset to 500 at
midnight, dozens of calls should have gone through before hitting the
ceiling again. This happened on both attempted resumptions.

**Working theory (not confirmable without AI Studio dashboard access):**
the free-tier quota likely refills gradually (a leaky-bucket/trickle model)
rather than jumping to a full 500 at a fixed daily boundary. This would
explain both observations: a small number of calls succeeding right after
a long idle period, then immediate re-exhaustion once that small buffer is
spent, with the next usable slot arriving only after a further wait.

**Consequence:** finishing the remaining ~48 classify/generate calls plus
198 judge calls may require many small attempts spread across a longer
window than "wait until tomorrow," not a single resumed run. Documented
here rather than guessed at silently — the eval report's results section
states plainly how many examples are covered as of any given snapshot.

## Resolved: second API key (different Google account) finished the run

A second free-tier key was created to unblock this. First attempt: a new
key generated under the *same* Google account landed in the same default
project (`gen-lang-client-...`) and hit the identical exhausted quota
immediately — confirms quota really is scoped per-project, not per-key.
Second attempt: a key from a genuinely different Google account worked
immediately and had a full, untouched budget. Cleared the remaining ~250
calls (48 classify/generate + 198 judge, plus some re-verification calls)
in about 20 minutes with zero further quota errors. Final results are in
`REPORT.md` and `artifacts/eval_report.json`; the committed fast-mode
cache (`.cache/llm_cache_golden.jsonl`) now has all 808 real responses
needed to reproduce them instantly via `make eval-fast`.

## The AI golden-set labeler shows 0/6 agreement with real human labels

**Discovered:** with working quota again, ran the actual spot-check
promised in the golden-set-labeling section — pointed the AI labeler
(`pipeline/ai_label_golden_set.py`) at the same 6 messages the human
labeled, same rubric, and compared. Result
(`artifacts/label_agreement_spotcheck.json`): **0/6 intent agreement**,
3/6 (coin-flip) escalation agreement.

**Why this matters more than the "AI-generated labels" limitation already
disclosed:** that limitation was framed as a risk (correlated errors
*might* exist between the labeler and the system being graded). This spot
-check is direct evidence, not a risk — the labeler doesn't even
reproduce a human's own judgment on the exact same rubric and text a
majority of the time. Since AI-generated labels are 192/198 (97%) of the
golden set, every intent-classification number in `REPORT.md` §6 is
measured against ground truth whose self-consistency against real humans,
on the only checkable slice, is 0%.

**Decision:** disclose this as the report's leading limitation (`REPORT.md`
§9, item 1) rather than let the strong-looking main-system macro-F1
(0.830) stand unqualified. Not treated as a reason to discard the eval
run — the escalation-recall finding and the retrieval/generation
groundedness spot-checks don't depend on golden-set intent labels being
correct — but the intent-classification headline number specifically
needs this caveat every time it's cited.

## Scope expansion: multilingual support (beyond PRD's English-only decision)

**Requested directly, not assumed:** the PRD explicitly scoped this
project to English-only (PRD §7, out of scope). Asked directly whether to
extend to other languages, including Indian languages. Confirmed the
specific shape wanted: classify + reply in the customer's own language,
still grounded on the existing English-only historical precedents
(translated in-prompt) — not a full separate taxonomy/index per language,
and not merely detect-and-escalate non-English messages untouched.

**What changed, in order:**
1. Swapped the embedding model from `all-MiniLM-L6-v2` to
   `paraphrase-multilingual-MiniLM-L12-v2` (`.env`, `pipeline/taxonomy.py`)
   — multilingual coverage (50+ languages incl. Hindi and other Indian
   languages) while staying in the same lightweight sentence-transformers
   family, not a heavier model like LaBSE.
2. Added best-effort multilingual risk-keyword and human-request regex
   patterns to `pipeline/escalation.py` (Spanish, Portuguese, French,
   German, Hindi) alongside the existing English-only ones — otherwise a
   non-English safety-trigger message would only be caught by the
   confidence/similarity thresholds, not the hard escalation trigger.
3. Updated `pipeline/generate.py`'s prompt to instruct the model to draft
   in the same language as the customer message, translating the
   substance of the (still English-only) precedent replies rather than
   replying in English regardless of input language.
4. Re-ran the full offline pipeline (embed → cluster → apply_taxonomy →
   split → index → baselines) with the new embedding model on all 103,757
   messages, and manually re-read `artifacts/cluster_samples.md` to remap
   the 27 new raw clusters onto the 8 existing named intents (`taxonomy.yaml`).

**Confirmed the intended effect, not just that it runs:** in the new
clustering, non-English messages about a known issue land in the same
cluster as the English reports of that issue — e.g. the iOS 11
WiFi/Bluetooth-re-enabling-itself cluster mixes English, German, and
Spanish complaints; the App Store cluster includes a Portuguese message —
instead of a blanket "non-English = out_of_scope" bucket the old
English-only embedding space produced. End-to-end tested with a real
Spanish message ("Mi iPhone se calienta mucho después de la
actualización...") — classified `software_update_bug` at 0.99 confidence
and drafted a reply in Spanish; correctly escalated because retrieval
similarity (0.87) fell under the tuned 0.90 threshold, since precedents
remain English-only.

**Documented, not hidden, gap:** `billing_subscription` and
`general_complaint` did not reform as their own clusters in the
multilingual re-clustering (`min_cluster_size=25` not met, or the
multilingual embedding space simply groups that content differently).
Their `cluster_ids` are empty in `taxonomy.yaml` — the live LLM classifier
can still assign either intent from the taxonomy description, but the
retrieval index currently has no precedent threads pre-labeled with them,
so grounded-reply quality for those two intents is weaker than for the
others. Not evaluated against a non-English golden set — the existing 198
-example golden set is still all-English, so this feature's classification
/generation quality is verified by spot-check and pipeline correctness,
not by a metric like the English-only eval numbers in `REPORT.md`.
