# Design Brief — Demo Frontend

## 1. Intent

This UI has one job: make a skeptical technical reviewer trust the system in under two minutes of looking at it. That means restraint, not spectacle. The system's rigor (grounded replies, honest escalation reasons, an eval dashboard that shows baselines and failure modes without flinching) is the actual selling point — the UI's job is to get out of the way and present that clearly, with enough polish that it doesn't undercut the engineering.

**Explicitly avoid:** cyan/purple gradient SaaS-template look, glowing neon accents, bouncy/flashy scroll animations, glassmorphism-for-its-own-sake, stock "AI product" iconography (glowing brains, sparkle emojis, chat-bubble robots). If it looks like a hundred other AI-wrapper landing pages, it's wrong.

**Aim for:** the visual register of a well-designed developer tool or editorial site — think Linear, Vercel, Stripe docs, or a well-typeset research report. Confident, quiet, a little austere. Class through restraint, not through decoration.

## 2. Visual System

### 2.1 Color
- Base: near-white or true off-white background (`#FAFAF8`–`#FFFFFF` range) in light mode, or a deep charcoal/near-black (`#121212`–`#181818`) if going dark-mode-first — pick one as primary, don't half-build both.
- Text: near-black/near-white, not pure `#000`/`#FFF` — slightly softened for reduced contrast harshness.
- **One** accent color, used sparingly (links, active states, the escalation-decision badge) — a muted, desaturated tone (deep ink blue, forest green, or warm graphite — explicitly *not* bright cyan, electric purple, or neon anything).
- Escalation states get semantic but muted color: auto-handle = quiet green/neutral, escalate = muted amber/red — never loud.

### 2.2 Typography
- One well-made grotesk/sans for UI text (Inter, Söhne, General Sans, or system stack done well) — don't mix more than two families.
- Optionally, a serif for the Report/long-form eval-writeup view only, to visually distinguish "reading" surfaces from "interacting" surfaces.
- Generous line-height on body text (1.5–1.6), tight tracking on large headings, real type scale (not just `text-lg`/`text-xl` defaults) — this alone does most of the "looks expensive" work.

### 2.3 Layout
- Wide margins, generous whitespace — content should feel like it's breathing, not packed edge-to-edge.
- Grid-based, not centered-card-soup. Use a real column structure for the eval dashboard (metrics table, confusion matrix, failure examples) rather than stacking cards vertically.
- Max content width capped (~1100–1200px) even on large screens — don't let text/tables stretch full-bleed.

## 3. Key Screens

### 3.1 Landing / Overview
- One clear statement of what this is (brand, the three capabilities, and — prominently — the headline eval number *alongside* the "what's misleading about it" caveat, right up front). This single choice signals more credibility than any visual polish.
- A short, static architecture diagram (can literally be a clean SVG rendering of the ASCII diagram in `ARCHITECTURE.md`) — no auto-playing animation, just a well-drawn diagram.

### 3.2 Live Demo Panel
- Left: a picker of sample real customer tweets (pulled from the eval or holdout set) plus a free-text input.
- Right: a vertical trace of the pipeline running — intent (with confidence), the retrieved precedent(s) shown as small quoted reference cards (real historical brand replies, clearly labeled as precedent, not as the answer), the drafted reply, and the escalation decision with its reason string rendered as plain, readable prose — not a cryptic badge.
- This view should read like a trace/log a careful engineer would want to see, not a chat bubble UI.

### 3.3 Evaluation Dashboard
- The most important screen. A real table: rows = trivial baseline / simple baseline / main system, columns = the actual metrics (accuracy, macro-F1, escalation precision/recall). No cherry-picked single number front and center — show the comparison.
- Confusion matrix rendered cleanly (a real heatmap-style grid, muted color scale — not rainbow).
- A "Top 5 Failure Modes" section with real examples inline (input → system output → why it's wrong), laid out like case studies, not hidden in a collapsed accordion.
- The "What's misleading about this number" section gets equal visual weight to the headline metric — same type scale, same prominence, not a footnote.

## 4. Motion

- Motion should clarify sequence, not decorate. Acceptable: content fading/sliding in slightly (8–16px translate, 150–250ms, ease-out) as sections enter viewport on scroll, staggered by ~40–60ms for grouped items (e.g., the four pipeline trace steps appearing in order).
- Not acceptable: parallax, bouncing/elastic easing, scroll-jacking, autoplaying loops, spinning/pulsing decorative elements, animated gradients.
- The pipeline trace (§3.2) is the one place a slightly more deliberate reveal animation is earned — each stage (classify → retrieve → draft → decide) appearing in sequence as if you're watching it think, at a natural reading pace (400–600ms between stages), not instant, not slow.
- Respect `prefers-reduced-motion` — disable non-essential transitions when set.

## 5. Component Notes

- Buttons/inputs: sharp-ish corners (4–8px radius, not full pill-everything), clear but quiet borders, no heavy drop shadows — a single subtle shadow at most for elevated surfaces (cards, modals).
- Code/data (retrieval scores, confidence numbers, reason strings) in a monospace font at small size — this is a strong, cheap signal of technical seriousness.
- Badges (intent label, auto-handle/escalate) as small, muted, text-forward pills — not oversized colorful chips.

## 6. Do / Don't

**Do:** whitespace, real typography scale, muted single-accent color, monospace for technical values, honest data visualization, quiet motion that clarifies sequence, dark-or-light chosen deliberately and executed fully.

**Don't:** cyan/purple SaaS gradients, glowing/neon anything, bouncy easing, scroll-jacking, stock AI iconography, cherry-picked metrics front-and-center without the caveat beside them, cramming the eval dashboard into an afterthought tab.
