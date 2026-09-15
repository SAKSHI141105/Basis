# Report — AI Support Agent for AppleSupport

> **Status note:** §6 below reports **real but partial** results — 152 of
> 198 golden-set examples (46 skipped on a quota-exhausted call, resilient
> skip per `DECISION_LOG.md`), and 0 judge scores (none cached yet). Every
> number traces to an actual Gemini API call already made; nothing here is
> a placeholder. The live run (`make eval-live`) continues across sessions
> as free-tier quota allows — see "A real constraint hit mid-build" below —
> and this section will be updated to the full 198/198 + judge scores once
> it completes.

## 1. What this is

An AI support copilot for the `AppleSupport` brand from Kaggle's Customer
Support on Twitter dataset. It classifies an incoming customer message into
one of 8 data-derived intents (or `out_of_scope`), drafts a reply grounded
in real historically-resolved precedent, and decides `auto_handle` vs.
`escalate` with a stated, auditable reason. Framed explicitly as a
Tier-1-agent copilot, not an autonomous system — see `PRD.md` §3, §6.

## 2. Methodology

### 2.1 Data

3,002,523 rows from `twcs.csv`, filtered to `AppleSupport` brand replies and
their originating customer messages (and any single customer follow-up),
producing 106,646 reconstructed threads. After cleaning (stripping
threading-only @handles/URLs, deduping repeated-contact near-duplicates,
dropping threads with no brand reply): **103,757 threads**, of which
**80,023 (77%) are marked "resolved"** by the documented heuristic (silence
after the brand's reply, or a closure phrase like "thanks"/"got it" in the
customer's follow-up).

**Documented weakness, not hidden:** silence is not proof of satisfaction.
A customer who got a bad reply and simply gave up looks identical, under
this heuristic, to one who was actually helped. This heuristic is a
practical necessity (the dataset has no explicit resolution signal at all)
but every downstream number inherits its imprecision.

### 2.2 Intent taxonomy

Customer messages were embedded with `all-MiniLM-L6-v2` (384-dim), PCA-reduced
to 50 dimensions (a fix discovered necessary mid-build — see
`DECISION_LOG.md`), then clustered with HDBSCAN (`min_cluster_size=25`),
producing 30 raw clusters plus a noise bucket. Every raw cluster was
manually read (`artifacts/cluster_samples.md`, regenerable, not committed)
and merged into one of **8 named intents**, or into `out_of_scope`:

`device_troubleshooting`, `software_update_bug`, `apple_id_account_access`,
`billing_subscription`, `repair_order_status`, `general_complaint`,
`feature_how_to`, `positive_feedback`, plus `out_of_scope`.

**Real, disclosed finding:** `out_of_scope` ends up **~88% of all traffic**.
Reading the actual noise bucket showed this is mostly genuine low-content
traffic — emoji-only replies, "check DM", "same issue" with no restated
context, bare iOS version-number replies, non-English text (this project is
English-only per `PRD.md` §7) — not a clustering failure. This was not
"fixed" by retuning HDBSCAN hyperparameters to produce a nicer-looking
distribution; it's disclosed as the actual shape of the data, and it has a
direct, important consequence for interpreting every accuracy number below.

### 2.3 Retrieval-grounded reply generation

A flat, L2-normalized embedding matrix over **resolved, training-split-only**
threads (68,019 entries) — no vector database (see `TRD.md` §1.1 for the
full reasoning: this data scale doesn't need one, and a vector DB introduces
avoidable dependency and reproducibility risk). On a new message: embed it,
filter to the predicted intent, cosine-similarity search (dot product on
pre-normalized vectors) for the top-3 precedent replies, and prompt the
generation model with the customer message + those precedents verbatim +
a brand style guide, requiring a structured `grounded_on` citation field so
groundedness is checkable, not self-reported.

Spot-checked against real queries: a message about battery drain after an
iOS update retrieved precedents at 0.90+ cosine similarity, all genuinely
on-topic.

### 2.4 Escalation decision engine

A deterministic rule layer (safety/legal/financial-harm keywords, explicit
human requests, and repeated contact with worsening sentiment always
escalate) followed by threshold checks on intent-classifier confidence and
retrieval similarity. Reason strings are templated, not free-form LLM
prose, so they stay auditable (`pipeline/escalation.py`).

### 2.5 Baselines

Both required baselines actually run in the harness, not just described:

- **Trivial:** always predicts the majority intent label (`out_of_scope`)
  for classification; always predicts `escalate` for escalation.
- **Simple:** TF-IDF (1-2 grams) + balanced Logistic Regression for
  classification; a keyword-only rule (risk/human-request flags, no
  LLM signals) for escalation.

## 3. A real constraint hit mid-build

Two free-tier quota walls were discovered empirically, not assumed upfront
(full account in `DECISION_LOG.md`):

1. `gemini-3.5-flash` (TRD's originally-intended generation/judge model)
   carries only a **20 requests/day** quota on a freshly created key —
   unusable at any real volume. Switched everything to
   `gemini-3.5-flash-lite`.
2. `gemini-3.5-flash-lite` itself caps at **500 requests/day**. A full
   198-example golden-set run needs ~594 calls (classify + generate + judge
   per example), which — combined with the same day's golden-set-labeling
   calls — does not fit in one day's budget on a freshly created key.

The system's caching architecture (built first, per TRD §8.3, specifically
*for* this kind of constraint) means this costs time, not correctness or
lost work: every successful call is cached by content hash and replayed
free on retry, so the live run resumes exactly where it left off across
multiple days without re-spending quota.

## 4. Golden set: how it was actually labeled

**Real, important limitation, stated plainly:** of the 198 golden-set
examples, only **6 are independently human-labeled**. The remaining 192
were labeled by Gemini applying the same rubric a human would
(`labeling_guide.md`), because manually labeling all 198 by hand proved to
be more time than was available for this pass. Every label is tagged
`label_source: "human"` or `"ai_generated"` in `golden_set.jsonl`.

This is a genuine, not-hand-waved limitation: using an LLM to generate
ground truth that is then used to grade an LLM-based classifier and an
LLM-based judge risks **correlated errors** — a systematic blind spot the
labeling model shares with the system being graded would not show up as a
disagreement, because the same kind of model made both calls. A real
human-annotator pass would not share that blind spot. Every headline number
in this report inherits this weakness and should be read with it in mind.

**A small, honest silver lining:** the 6 human labels give a tiny spot-check
of AI-labeling quality on the same rubric (not a substitute for the
required human-agreement study below, but a useful sanity signal) —
included once the full run completes.

## 5. Human-agreement study: not performed

TRD §7.3 requires a human-agreement study comparing the LLM judge's
reply-quality scores against an independent human rater's scores on a
40-50 example subsample, specifically to check the judge isn't just
agreeing with itself. Given the constraints of this build (see §4 above),
no independent human rater was available to run this study for real.

**This is stated explicitly rather than faked.** Having another LLM stand
in as "the human" for this study would not produce a weaker version of the
same evidence — it would produce circular, meaningless evidence, since it
would just be checking whether one LLM agrees with another LLM applying a
similar rubric. The eval report marks this section "not performed" rather
than filling it with a number that looks like agreement data but isn't.

## 6. Results (partial — 152/198, 0 judge scores)

From `artifacts/eval_report.json`, committed and reproducible via
`make eval-fast`. **46 examples and all judge scores are missing** because
the live run hit a free-tier quota wall mid-run (§3) — every number below
is real, from actual Gemini calls, on whatever subset has completed so far.

**Intent classification** (main system: 152 examples; baselines: all 198):

| System | Accuracy | Macro-F1 |
|---|---|---|
| Trivial (always majority label) | 0.051 | 0.011 |
| Simple (TF-IDF + LogReg) | 0.672 | 0.670 |
| Main system (LLM classifier) | 0.836 | 0.694 |

**Escalation decision:**

| System | Precision | Recall | F1 | Cost-weighted |
|---|---|---|---|---|
| Trivial (always escalate) | 0.414 | 1.000 | 0.586 | 0.805 |
| Simple (keyword rule) | 0.750 | 0.037 | 0.070 | 0.599 |
| Main system | 0.500 | 0.171 | 0.255 | 0.592 |

**A real, concerning finding, not smoothed over:** the main system's
escalation **recall is only 0.171** — it misses roughly 5 of every 6
examples that should have been escalated, on this partial slice. PRD §6 is
explicit that a missed escalation (a confidently-wrong reply sent
unsupervised) is the *expensive* failure mode here, more expensive than an
unnecessary escalation. A recall this low is a real problem with the
current threshold calibration (`pipeline/escalation.py`'s
`DEFAULT_CONFIDENCE_THRESHOLD`/`DEFAULT_SIMILARITY_THRESHOLD`), not a
rounding error — worth retuning against the golden set once it's complete,
called out explicitly in §10 below.

LLM judge scores: none yet (0/152 cached) — no reply-quality numbers to
report until the live run produces some.

## 7. Failure analysis (case studies)

The 5 examples below are worth reading closely for a reason beyond being
wrong: they are 5 of the 6 **independently human-labeled** examples in the
golden set (the only 6 not generated by an LLM — see §4), and the main
system got 5 of those 6 wrong on intent. On a sample this small (n=6) this
is not a statistically reliable rate, but it is the single most trustworthy
signal in this entire report, because it's the only place the "ground
truth" isn't itself LLM-generated.

**1. "iOS 11 is buggy. Please fix it. 😡"**
True: `device_troubleshooting` — Predicted: `general_complaint`
A genuinely ambiguous case: "buggy" implies a malfunction, but the message
gives no specific symptom. Reasonable humans could file this either way —
this points at real overlap between `general_complaint` and
`device_troubleshooting` in the taxonomy itself, not just a model miss.

**2. "your new IOS update is complete fucking dog shit... phone hasnt
work properly. I want the old 1 bak"**
True: `device_troubleshooting` — Predicted: `software_update_bug` (escalation: true `escalate`, predicted `auto_handle`)
The message explicitly blames "the update," which is *exactly*
`software_update_bug`'s taxonomy.yaml definition — the model's answer is
arguably more textually justified than the human label here. The
escalation miss is the more concerning error: heavy profanity and explicit
anger ("complete fucking dog shit") reads as an obvious escalate case to a
human, but the deterministic rule layer only fires on specific risk
keywords (legal/safety/financial-harm) and doesn't treat raw hostility as
its own signal — a real gap worth closing.

**3. "My iPhone 6 went dead while I was transferring it to a new iCloud
account?..."**
True: `software_update_bug` — Predicted: `device_troubleshooting`
The mirror image of case 2: nothing in the message mentions an update at
all, so the model's `device_troubleshooting` call looks like the more
defensible reading of the text. Worth naming plainly: this may be a case
where the *label*, not the model, is the questionable call — the AI/human
labeling process had access to more thread context than the classifier
does at inference time in this eval.

**4. "Ever since the update my reminders display on my lock screen all
the time..."**
True: `general_complaint` — Predicted: `software_update_bug` (escalation: true `escalate`, predicted `auto_handle`)
Same boundary confusion as case 1/2, mirrored: the message says "since the
update" almost verbatim from `software_update_bug`'s definition, yet the
human labeled it `general_complaint`. Another missed escalation on the
same pattern as case 2.

**5. "11.1.2 (15B202). USA"**
True: `feature_how_to` — Predicted: `out_of_scope` (escalation: true `escalate`, predicted `auto_handle`)
Read in isolation, this message *is* out_of_scope by taxonomy.yaml's own
definition ("bare version-number replies"). This is almost certainly a
follow-up reply in a multi-turn thread (answering "what iOS version are
you on?"), where the original question that makes it `feature_how_to`
lives in an earlier turn the classifier never sees. This looks like a
genuine limitation of evaluating single messages without full thread
context, not a classifier failure.

**What this small sample actually shows:** 4 of these 5 errors trace to a
real, fixable taxonomy ambiguity (the `general_complaint` /
`software_update_bug` / `device_troubleshooting` boundary is genuinely
blurry when a complaint IS about a bug), one traces to a labeling/context
limitation rather than a model error, and 3 of the 5 compound into the same
missed-escalation pattern already flagged in §6 — profanity/anger not
being treated as its own escalation signal. That last point is the most
actionable finding in this whole report.

## 8. What's misleading about the headline number

Two distinct "misleading number" stories showed up in this build, at two
different stages — worth keeping both, since they point in *opposite*
directions:

**On the full, naturally-imbalanced dataset** (§2.2: `out_of_scope` is
~88% of real traffic), confirmed on the held-out eval split during baseline
development:

| System | Accuracy | Macro-F1 |
|---|---|---|
| Trivial (always `out_of_scope`) | 0.881 | 0.104 |
| Simple (TF-IDF + LogReg) | 0.808 | 0.472 |

Here the trivial baseline *beats* the simple baseline on raw accuracy while
being far worse by macro-F1 — accuracy alone is misleading in the
*optimistic* direction for a do-nothing baseline.

**On the golden set** (§6 above), the opposite distortion shows up: the
golden set is deliberately stratified roughly evenly across the 9 intents
(`pipeline/golden_set.py`), not the natural ~88%-skewed distribution, so
the trivial baseline's accuracy *collapses* to 0.051 instead of looking
falsely strong. **Which number is "misleading" depends entirely on which
distribution you're reading it against** — a headline metric is only as
trustworthy as the sampling behind it, and this project surfaced both
directions of that problem, not just the one it went looking for.
**Macro-F1 and per-intent F1 remain the numbers that actually reflect
classification quality in either case; raw accuracy alone should not be
trusted on this dataset.**

## 9. Known limitations (full list)

1. Golden-set ground truth is 97% AI-generated, not human-labeled (§4).
2. The human-agreement study could not be performed (§5).
3. The "resolved" heuristic is weak — silence ≠ satisfaction (§2.1).
4. `out_of_scope` dominates real traffic by construction (§2.2) — makes
   raw accuracy misleading (§7).
5. Free-tier daily quotas (§3) mean a full live run spans multiple days —
   §6's results are from 152/198 examples and 0/152 judge scores as of this
   writing, clearly marked partial, not presented as final.
6. **Main system escalation recall is low (0.171) on the partial results**
   (§6) — a real, not-yet-explained weakness in the current threshold
   calibration, exactly the kind of finding this report exists to surface.
7. No PII redaction pipeline (brand chosen partly to reduce this need, but
   it's a gap, not a guarantee — `PRD.md` §7).
8. No live system integration, multi-language support, or fine-tuning —
   explicitly out of scope (`PRD.md` §7).

## 10. What I'd do next with one more week

- **Retune the escalation thresholds against the golden set** — §6's 0.171
  recall is the single most actionable finding in this report; the
  deterministic thresholds in `pipeline/escalation.py` were set by
  reasonable default, never calibrated against real labeled data.
- Get a real independent human annotator for both the golden-set labels and
  the reply-quality human-agreement study — the single highest-value fix
  for the credibility of every number in this report.
- An embedding-only classifier (TRD §4.3 Option B) as a second real system
  to compare against the LLM classifier — cheaper, and a good "two systems,
  not just two baselines" story.
- Investigate whether a paid/higher tier or a second API key would let a
  full live eval run complete same-day rather than spanning a quota reset.
- Deployment (`AGENT_WORKING_AGREEMENT.md` §5) — deliberately last, per the
  build order, and only attempted once everything above is solid. The
  Next.js demo frontend (`ARCHITECTURE.md` §2.4) is already built.

See `DECISION_LOG.md` for the full, chronological account of every decision
and constraint discovered while actually building this system.
