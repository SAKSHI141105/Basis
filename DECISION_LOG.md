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
