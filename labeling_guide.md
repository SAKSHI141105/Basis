# Golden Set Labeling Guide

This documents the rubric used to hand-label `golden_set.jsonl` (TRD 7.1),
so the method is reproducible even though the label itself is a subjective
judgment call made by one author.

## What gets labeled per example

For each sampled thread:

1. **`true_intent`** — one of the ids in `taxonomy.yaml`. If the message
   genuinely straddles two intents, pick the one the *first* sentence is
   about (matches how a real Tier-1 agent would triage on first read).
2. **`true_escalation`** — `auto_handle` or `escalate`, using the same
   judgment a competent Tier-1 agent would apply: would you be comfortable
   auto-sending a reply to this customer with no further human review?
   - Escalate if: the issue needs account-specific action the brand can't
     do in a public/templated reply, the customer is visibly upset after
     repeated contact, there's any safety/legal/financial-harm language, or
     the "right" reply isn't obvious from any precedent in the data.
   - Auto-handle if: the issue is a known, templatable fix (password reset
     link, standard troubleshooting step, a FAQ-style answer) and the
     customer's tone doesn't suggest they're already frustrated.
3. **`ambiguous_note`** (optional, freeform) — a short note *only* when the
   "right" answer isn't clear-cut. Left blank means "this one was clear."
   These notes are what the report's failure-analysis section draws from.

## Sampling method (for reference — see `pipeline/golden_set.py`)

- Stratified by derived intent (roughly equal quota per intent).
- Within each intent's quota, ~30% deliberately oversampled from "hard"
  cases: multi-turn threads, messages containing an escalation-trigger-like
  keyword, or unusually short messages — so the golden set isn't dominated
  by easy, obviously-auto-handleable traffic (PRD risk table).
- Fixed random seed (see `pipeline/config.py: RANDOM_SEED`) for
  reproducibility of the sample itself, independent of the labels applied
  to it.

## Known limitation of this method

One author labels the entire set in one pass — there is no second labeler
and no adjudication step for disagreements, because a one-week take-home
budget doesn't support a full annotation study. This is a real limitation:
the "true" labels reflect one person's judgment, not inter-annotator
consensus. It's disclosed here and in the report's "what's misleading about
my headline number" section, not hidden.

The **separate** human-agreement study (TRD 7.3) is not a substitute for
this — that study checks whether the *LLM judge* agrees with a human on
*reply-quality* scores, not whether two humans agree on intent/escalation
ground truth.
