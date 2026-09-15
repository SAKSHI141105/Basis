import Link from "next/link";
import { getEvalReport } from "@/lib/api";
import { Reveal } from "@/components/Reveal";

const CAPABILITIES = [
  {
    index: "01",
    title: "Classify",
    body: "Sorts an incoming message into one of 8 data-derived intents (or out_of_scope) with a confidence score, using a taxonomy frozen from real clustering — not invented top-down.",
  },
  {
    index: "02",
    title: "Draft, grounded",
    body: "Retrieves the most similar historically-resolved AppleSupport replies and drafts a response conditioned on them, citing exactly which precedent it used.",
  },
  {
    index: "03",
    title: "Decide, out loud",
    body: "Auto-handles or escalates to a human, with a templated, auditable reason string — never a bare label.",
  },
];

export default async function OverviewPage() {
  const report = await getEvalReport().catch(() => null);

  return (
    <div className="relative">
      <div className="hero-glow absolute inset-x-0 top-0 h-[420px] pointer-events-none" />
      <div className="relative mx-auto max-w-[1120px] px-6 py-20">
        <Reveal>
          <p className="font-mono text-[12px] tracking-[0.08em] uppercase text-text-muted mb-6">
            AppleSupport &middot; AI support agent
          </p>
          <h1 className="font-display text-[44px] leading-[1.12] font-medium tracking-tight max-w-[760px] text-balance text-text">
            A copilot that classifies, drafts a grounded reply, and decides
            whether a human needs to step in.
          </h1>
          <p className="mt-6 max-w-[600px] text-[16px] leading-[1.65] text-text-secondary">
            Built on real AppleSupport support threads from Twitter. Every claim
            on this page is checkable: the taxonomy came from clustering real
            messages, every drafted reply cites a real precedent, and every
            escalation decision comes with a plain-English reason.
          </p>

          <div className="mt-9 flex gap-3">
            <Link
              href="/demo"
              className="rounded-md bg-accent px-5 py-2.5 text-[13.5px] font-medium text-white shadow-[var(--shadow-sm)] transition-all hover:bg-accent-hover hover:shadow-[var(--shadow-md)] hover:-translate-y-px"
            >
              Try the live demo
            </Link>
            <Link
              href="/eval"
              className="rounded-md border border-border bg-surface px-5 py-2.5 text-[13.5px] font-medium text-text transition-all hover:border-border-strong hover:-translate-y-px"
            >
              See the eval numbers
            </Link>
          </div>
        </Reveal>

        <section className="mt-24 grid grid-cols-1 md:grid-cols-3 gap-5">
          {CAPABILITIES.map((c, i) => (
            <Reveal key={c.title} delay={i * 90}>
              <div className="hover-lift rounded-xl border border-border bg-surface p-7 shadow-[var(--shadow-card)] h-full">
                <p className="font-mono text-[11px] text-text-muted mb-4">{c.index}</p>
                <h3 className="font-display text-[19px] font-medium mb-2.5 text-text">{c.title}</h3>
                <p className="text-[13.5px] leading-[1.65] text-text-secondary">{c.body}</p>
              </div>
            </Reveal>
          ))}
        </section>

        <Reveal className="mt-24">
          <section>
            <div className="flex items-baseline justify-between mb-5">
              <h2 className="text-[12.5px] font-medium uppercase tracking-[0.08em] text-text-muted">
                Headline number
              </h2>
              <Link
                href="/eval"
                className="text-[12.5px] font-mono text-accent hover:text-accent-hover transition-colors"
              >
                full dashboard &rarr;
              </Link>
            </div>

            {report ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                <div className="hover-lift rounded-xl border border-border bg-surface p-7 shadow-[var(--shadow-card)]">
                  <p className="font-mono text-[11.5px] text-text-muted mb-2">main system &middot; macro-F1</p>
                  <p className="font-display text-[42px] font-medium tabular-nums text-text">
                    {report.intent_metrics.main.macro_f1.toFixed(3)}
                  </p>
                  <p className="mt-3 text-[13.5px] text-text-secondary">
                    vs.{" "}
                    <span className="font-mono tabular-nums">
                      {report.intent_metrics.trivial.macro_f1.toFixed(3)}
                    </span>{" "}
                    trivial baseline,{" "}
                    <span className="font-mono tabular-nums">
                      {report.intent_metrics.simple.macro_f1.toFixed(3)}
                    </span>{" "}
                    TF-IDF baseline
                  </p>
                </div>
                <div className="hover-lift rounded-xl border border-warning-border bg-warning-bg p-7 shadow-[var(--shadow-card)]">
                  <p className="font-mono text-[11px] uppercase tracking-[0.05em] text-warning mb-2">
                    what&apos;s misleading about this
                  </p>
                  <p className="text-[13.5px] leading-[1.65] text-text">{report.misleading_number_note}</p>
                </div>
              </div>
            ) : (
              <div className="rounded-xl border border-border bg-surface-2 p-7 text-[13.5px] leading-[1.65] text-text-secondary font-mono">
                eval_report.json not yet generated — run <code>make eval-fast</code> or{" "}
                <code>make eval-live</code>, or open the{" "}
                <Link href="/eval" className="text-accent underline underline-offset-2">
                  eval dashboard
                </Link>{" "}
                for details.
              </div>
            )}
          </section>
        </Reveal>

        <Reveal className="mt-24 mb-8">
          <section>
            <Link
              href="/docs"
              className="inline-flex items-center gap-1.5 text-[13.5px] font-mono text-accent hover:text-accent-hover transition-colors"
            >
              How it&apos;s built &mdash; architecture &amp; docs &rarr;
            </Link>
          </section>
        </Reveal>
      </div>
    </div>
  );
}
