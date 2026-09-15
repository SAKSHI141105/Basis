import Link from "next/link";
import { ArchitectureDiagram } from "@/components/ArchitectureDiagram";

const DOCS = [
  {
    file: "PRD.md",
    title: "Product requirements",
    body: "What this agent is for, who it's for, and what's explicitly out of scope.",
  },
  {
    file: "TRD.md",
    title: "Technical requirements",
    body: "The taxonomy, retrieval, escalation, and evaluation methodology this system commits to.",
  },
  {
    file: "ARCHITECTURE.md",
    title: "Architecture",
    body: "Why offline pipeline / online serving is split the way it is, and why there's no vector DB at this scale.",
  },
  {
    file: "REPORT.md",
    title: "Evaluation report",
    body: "Every real result, every baseline, every disclosed limitation and misleading number.",
  },
  {
    file: "DECISION_LOG.md",
    title: "Decision log",
    body: "The chronological, real account of every constraint hit and decision made while building this.",
  },
];

export default function DocsPage() {
  return (
    <div className="mx-auto max-w-[1120px] px-6 py-16">
      <p className="font-mono text-[12px] tracking-[0.08em] uppercase text-text-muted mb-3">
        Documentation
      </p>
      <h1 className="font-display text-[32px] leading-[1.2] font-medium text-text max-w-[640px]">
        How this system is built, and why.
      </h1>
      <p className="mt-4 max-w-[640px] text-[14.5px] leading-[1.65] text-text-secondary">
        Moved off the overview page to keep that page focused on what the
        agent does. This page is for anyone who wants the reasoning behind
        it.
      </p>

      <section className="mt-14">
        <h2 className="text-[12.5px] font-medium uppercase tracking-[0.08em] text-text-muted mb-5">
          Architecture
        </h2>
        <div className="rounded-xl border border-border bg-surface p-8 shadow-[var(--shadow-card)]">
          <ArchitectureDiagram />
        </div>
      </section>

      <section className="mt-14 mb-8">
        <h2 className="text-[12.5px] font-medium uppercase tracking-[0.08em] text-text-muted mb-5">
          Source documents (in the repo root)
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {DOCS.map((d) => (
            <div
              key={d.file}
              className="rounded-xl border border-border bg-surface p-6 shadow-[var(--shadow-card)]"
            >
              <p className="font-mono text-[11px] text-accent mb-2">{d.file}</p>
              <h3 className="font-display text-[16px] font-medium mb-1.5 text-text">{d.title}</h3>
              <p className="text-[13px] leading-[1.6] text-text-secondary">{d.body}</p>
            </div>
          ))}
        </div>
      </section>

      <Link
        href="/"
        className="inline-flex items-center gap-1.5 text-[13.5px] font-mono text-accent hover:text-accent-hover transition-colors"
      >
        &larr; Back to overview
      </Link>
    </div>
  );
}
