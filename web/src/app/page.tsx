import Link from "next/link";
import { ArchitectureDiagram } from "@/components/ArchitectureDiagram";
import { getEvalReport } from "@/lib/api";

const CAPABILITIES = [
  {
    title: "Classify",
    body: "Sorts an incoming message into one of 8 data-derived intents (or out_of_scope) with a confidence score, using a taxonomy frozen from real clustering — not invented top-down.",
  },
  {
    title: "Draft, grounded",
    body: "Retrieves the most similar historically-resolved AppleSupport replies and drafts a response conditioned on them, citing exactly which precedent it used.",
  },
  {
    title: "Decide, out loud",
    body: "Auto-handles or escalates to a human, with a templated, auditable reason string — never a bare label.",
  },
];

export default async function OverviewPage() {
  const report = await getEvalReport().catch(() => null);

  return (
    <div className="mx-auto max-w-[1120px] px-6 py-16">
      <p className="font-mono text-[12px] text-text-muted mb-4">AppleSupport &middot; AI support agent</p>
      <h1 className="text-[32px] leading-[1.15] font-medium tracking-tight max-w-[720px] text-balance">
        A copilot that classifies, drafts a grounded reply, and decides whether a
        human needs to step in.
      </h1>
      <p className="mt-5 max-w-[640px] text-[15px] leading-[1.6] text-text-secondary">
        Built on real AppleSupport support threads from Twitter. Every claim on this
        page is checkable: the taxonomy came from clustering real messages, every
        drafted reply cites a real precedent, and every escalation decision comes
        with a plain-English reason.
      </p>

      <div className="mt-8 flex gap-3">
        <Link
          href="/demo"
          className="rounded-md bg-accent px-4 py-2 text-[13px] font-medium text-white"
        >
          Try the live demo
        </Link>
        <Link
          href="/eval"
          className="rounded-md border border-border px-4 py-2 text-[13px] font-medium text-text hover:border-border-strong"
        >
          See the eval numbers
        </Link>
      </div>

      <section className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-px bg-border rounded-lg overflow-hidden border border-border">
        {CAPABILITIES.map((c) => (
          <div key={c.title} className="bg-surface p-6">
            <h3 className="text-[14px] font-medium mb-2">{c.title}</h3>
            <p className="text-[13.5px] leading-[1.6] text-text-secondary">{c.body}</p>
          </div>
        ))}
      </section>

      <section className="mt-16">
        <div className="flex items-baseline justify-between mb-4">
          <h2 className="text-[13px] font-medium uppercase tracking-[0.06em] text-text-muted">
            Headline number
          </h2>
          <Link href="/eval" className="text-[12px] font-mono text-accent">
            full dashboard &rarr;
          </Link>
        </div>

        {report ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-px bg-border rounded-lg overflow-hidden border border-border">
            <div className="bg-surface p-6">
              <p className="text-[12px] font-mono text-text-muted mb-1">main system &middot; macro-F1</p>
              <p className="text-[36px] font-mono font-medium tabular-nums">
                {report.intent_metrics.main.macro_f1.toFixed(3)}
              </p>
              <p className="mt-2 text-[13px] text-text-secondary">
                vs. {report.intent_metrics.trivial.macro_f1.toFixed(3)} trivial baseline,{" "}
                {report.intent_metrics.simple.macro_f1.toFixed(3)} TF-IDF baseline
              </p>
            </div>
            <div className="bg-warning-bg p-6">
              <p className="text-[12px] font-mono uppercase tracking-[0.04em] text-warning mb-1">
                what&apos;s misleading about this
              </p>
              <p className="text-[13.5px] leading-[1.6] text-text">{report.misleading_number_note}</p>
            </div>
          </div>
        ) : (
          <div className="rounded-lg border border-border bg-surface-2 p-6 text-[13.5px] text-text-secondary font-mono">
            eval_report.json not yet generated — run <code>make eval-fast</code> or{" "}
            <code>make eval-live</code>, or open the{" "}
            <Link href="/eval" className="text-accent underline">
              eval dashboard
            </Link>{" "}
            for details.
          </div>
        )}
      </section>

      <section className="mt-16">
        <h2 className="text-[13px] font-medium uppercase tracking-[0.06em] text-text-muted mb-4">
          Architecture
        </h2>
        <div className="rounded-lg border border-border bg-surface p-6">
          <ArchitectureDiagram />
        </div>
      </section>
    </div>
  );
}
