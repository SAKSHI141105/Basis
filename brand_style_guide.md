# AppleSupport Brand Style Guide

Derived from 20 real, resolved AppleSupport replies (`pipeline/derive_style_guide.py`,
`seed=42` for reproducibility — TRD §5.3). Not assumed; measured.

## Measured patterns (n=20)

- **Length:** 14–40 words, mean ~22 words. Short — one to three sentences.
- **Collaborative framing:** 13/20 use "we" ("We're here to help," "Let's
  look into this together") — never "I."
- **Defers account-specific detail to DM:** 9/20 explicitly direct the
  customer to DM for anything requiring personal/device detail — the
  public reply rarely tries to solve the issue inline when it needs
  account-specific info.
- **Rarely asks a clarifying question in the same breath as helping:**
  only 3/20 ask a question outright; more often the reply gives a step or
  a link, or invites DM, rather than interrogating the customer publicly.
- **No sign-off pattern observed** (no agent initials or signature) in this
  brand's replies, unlike some support accounts — a real, checked absence,
  not an oversight.
- **No apology-heavy or corporate-filler language** in the sample — no
  "we sincerely apologize for the inconvenience" boilerplate.

## The guide (used verbatim in `service/state.py`)

> Warm but concise (aim for 1–3 short sentences, ~15–40 words). Use "we,"
> never "I." Acknowledge the issue in one short clause, then either give a
> concrete next step/link or invite the customer to DM for anything
> requiring account-specific detail — don't try to solve account-specific
> issues in the public reply. No sign-off/initials. No apology-heavy or
> corporate filler language.
