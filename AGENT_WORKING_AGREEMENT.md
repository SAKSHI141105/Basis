# Agent Working Agreement — Process, Git Hygiene & Deployment

This document is instructions **for the coding agent**, not product spec. Follow it alongside `PRD.md`, `TRD.md`, and `ARCHITECTURE.md`. It governs *how* to build this, not *what* to build.

## 1. Do Not Build This In One Shot

Build incrementally, in the milestone order given in `PRD.md` §9 (data pipeline → taxonomy → baselines → retrieval/generation → escalation → eval harness → report → frontend → deployment). At the end of each meaningful unit of work, stop, verify it actually runs, commit it, and push it — then move to the next unit. A single giant commit at the end is a failure of this instruction even if the final code is correct, because it gives no reviewable trail of decisions.

**A "meaningful unit of work" is roughly:** one pipeline stage working end-to-end on real (even if small/sample) data, one API endpoint working with a real request/response, one test suite passing, one section of the eval harness producing real output, one frontend view rendering real data. If you can describe what you just did in one sentence and it either works or clearly doesn't, that's commit-sized.

## 2. Commit Discipline

- **Commit early, commit often.** Roughly: after ingestion works, after cleaning works, after taxonomy is frozen, after each baseline, after each of the three core endpoints, after the eval harness produces a report, after each major report section is drafted, after each frontend view, after deployment configuration. That's on the order of 20–35 commits for the whole project, not 3–5.
- **Commit messages: short, crisp, to the point.** Conventional-commit style is preferred but not mandatory — what matters is that someone scanning `git log --oneline` understands the project's build order without opening a single diff.
  - Good: `feat: reconstruct customer/brand thread pairs from twcs.csv`
  - Good: `fix: dedupe near-identical repeated-contact messages before clustering`
  - Good: `feat: add retrieval-grounded reply generation with precedent citation`
  - Good: `test: add fixture-based unit tests for thread reconstruction`
  - Bad: `updates` / `wip` / `final version` / `fix stuff` / a 200-word commit body explaining the whole architecture.
- **Never commit secrets.** `.env` stays gitignored; only `.env.example` (with placeholder values) is committed. Double-check this before every push, not just once at setup.
- **Never commit large/regenerable artifacts.** `artifacts/`, `.cache/`, model weights, and the full raw dataset are gitignored — the README's reproduction steps regenerate them. The one exception is the small precomputed LLM-response cache used for the fast eval mode (TRD §8.2) — that one is intentionally committed because it *is* the proof artifact.
- **Push after every commit, or at minimum after every milestone**, so there's always a recoverable, inspectable remote history — don't accumulate a day's worth of local-only commits.

## 3. Branching (keep it simple)

A take-home doesn't need a full branching model, but don't do everything on a single branch with no structure either:
- `main` stays in a working state at all times — if something is committed to `main`, it runs.
- For any change larger than one milestone unit (e.g., "the whole eval harness"), use a short-lived feature branch, then merge to `main` once it runs end-to-end, with a single clean merge commit message summarizing what landed.
- Small, self-contained fixes can go straight to `main`.

## 4. Definition of Done for Each Milestone

Before committing a milestone as complete, confirm:
1. It runs from a clean state (no reliance on manual steps you forgot to script).
2. It has at least a minimal test or a manual sanity check documented in the commit message or a comment.
3. The README is updated in the same commit if the setup/run steps changed — README drift is a common failure mode and should be treated as part of the work, not cleanup for later.

## 5. Deployment — Full Walkthrough (do this last, only once the pipeline + eval harness + report are done and reproducible)

The core grading criteria (README reproducibility, golden set, eval harness, report, decision log) do **not** require a live deployment — a local, reproducible repo satisfies them. A live deployment is the polish on top (matches the Design Brief's demo frontend), and should only be attempted after everything else is solid, so a deployment issue can never put the core deliverables at risk.

### 5.1 Backend (FastAPI) → Render (free tier)
1. Add a `Dockerfile` (or Render's native Python runtime, either works — Docker is more portable/reproducible, prefer it) that installs `requirements.txt` and runs `uvicorn service.main:app --host 0.0.0.0 --port $PORT`.
2. Push the repo to GitHub (required — Render deploys from a GitHub repo).
3. Create a free **Web Service** on Render, connect the GitHub repo, set the build/start commands, and add the environment variables from `.env.example` (`GEMINI_API_KEY`, model names) in Render's dashboard — never in the repo.
4. Confirm the health check endpoint (add a simple `GET /health` if one doesn't exist yet) responds before wiring up the frontend.
5. **Document the known limitation, don't hide it:** Render's free web services spin down after ~15 minutes of inactivity and take ~30–50 seconds to wake on the next request. Add a one-line note in both the README and the frontend itself ("waking up the backend — first request may take up to ~30s") so this reads as an accepted trade-off, not a bug.

### 5.2 Frontend (Next.js) → Vercel (free tier)
1. Set `NEXT_PUBLIC_API_URL` to the deployed Render backend URL as a Vercel environment variable (not hardcoded).
2. Connect the same GitHub repo to Vercel (it auto-detects Next.js — no custom build config needed in the common case).
3. Deploy, then verify the live demo panel and eval dashboard both correctly call the deployed backend (not `localhost`) — this is the most common last-mile bug, check it explicitly rather than assuming the env var wired through.
4. Vercel's free tier has no cold-start sleep issue for the frontend itself — only the Render backend call will show the wake-up delay.

### 5.3 Final Check Before Calling It Done
- Open the deployed frontend in an incognito window (no cached local state) and run through the live demo panel once, end to end, exactly as a reviewer would.
- Confirm the eval dashboard reflects the *committed* `eval_report.json` (the same one referenced in the written Report deliverable) — a live deployment showing different numbers than the written report is a credibility problem worth explicitly guarding against.
- Add the live URLs (frontend + backend health check) to the top of the README, clearly separated from the "reproduce it yourself locally" instructions — both paths should be available to the reviewer, not one instead of the other.

## 6. On the Optional Banking77 Extension

If pursued (see the project brainstorm notes for the feasibility call — it's a small, worthwhile stretch, not core scope): treat it strictly as a **methodology validation side-quest**, done after the core AppleSupport pipeline is fully working, using the *same* classifier code path against Banking77's labeled data as a sanity check that the classification approach is sound on data with known ground truth. It gets its own short subsection in the report, clearly marked optional/stretch, and must never delay or put at risk any of the five core deliverables in `PRD.md`. If time runs short, cut this first — it is explicitly the lowest-priority item in the whole plan.
