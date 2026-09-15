"use client";

import { useEffect, useState } from "react";
import { ApiError, PipelineResponse, SampleMessage, getSamples, runPipeline } from "@/lib/api";
import { ConfidenceValue, EscalationBadge, IntentBadge } from "@/components/Badges";

type Stage = "classify" | "retrieve" | "draft" | "decide";
const STAGES: { key: Stage; label: string }[] = [
  { key: "classify", label: "Classify" },
  { key: "retrieve", label: "Retrieve precedent" },
  { key: "draft", label: "Draft reply" },
  { key: "decide", label: "Decide" },
];

export default function DemoPage() {
  const [samples, setSamples] = useState<SampleMessage[]>([]);
  const [message, setMessage] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [result, setResult] = useState<PipelineResponse | null>(null);
  const [visibleStage, setVisibleStage] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getSamples()
      .then(setSamples)
      .catch(() => setSamples([]));
  }, []);

  async function handleRun() {
    if (!message.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setVisibleStage(0);
    try {
      const res = await runPipeline(message.trim());
      setResult(res);
      // reveal the trace stage by stage, at reading pace (Design Brief 4)
      for (let i = 1; i <= STAGES.length; i++) {
        await new Promise((r) => setTimeout(r, 450));
        setVisibleStage(i);
      }
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Request failed — is the API running?");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto max-w-[1120px] px-6 py-16 grid grid-cols-1 md:grid-cols-[380px_1fr] gap-12">
      <div>
        <h1 className="font-display text-[24px] font-medium mb-2 text-text">Live demo</h1>
        <p className="text-[13.5px] leading-[1.6] text-text-secondary mb-7">
          Pick a real customer message from the golden set, or write your own.
        </p>

        <div className="mb-7">
          <label className="block text-[11px] font-mono uppercase tracking-[0.06em] text-text-muted mb-2">
            Pick an example
          </label>
          <select
            value={selectedId ?? ""}
            onChange={(e) => {
              const s = samples.find((s) => s.thread_id === e.target.value);
              if (s) {
                setSelectedId(s.thread_id);
                setMessage(s.message);
              }
            }}
            disabled={samples.length === 0}
            className="w-full rounded-lg border border-border bg-surface px-3.5 py-3 text-[13px] text-text shadow-[var(--shadow-card)] disabled:opacity-50"
          >
            <option value="" disabled>
              {samples.length === 0 ? "No samples loaded" : "Select a golden-set example…"}
            </option>
            {samples.map((s) => (
              <option key={s.thread_id} value={s.thread_id}>
                {s.message.length > 80 ? s.message.slice(0, 80) + "…" : s.message}
              </option>
            ))}
          </select>
          {samples.length === 0 && (
            <p className="mt-2 text-[12px] font-mono text-text-muted">
              Is the API running at the configured URL?
            </p>
          )}
        </div>

        <textarea
          value={message}
          onChange={(e) => {
            setMessage(e.target.value);
            setSelectedId(null);
          }}
          placeholder="Or write a customer message here…"
          rows={4}
          className="w-full rounded-lg border border-border bg-surface px-3.5 py-3 text-[13.5px] leading-[1.5] text-text placeholder:text-text-muted resize-none shadow-[var(--shadow-card)]"
        />
        <button
          onClick={handleRun}
          disabled={loading || !message.trim()}
          className="mt-3 w-full rounded-lg bg-accent px-4 py-3 text-[13.5px] font-medium text-white shadow-[var(--shadow-sm)] transition-colors hover:bg-accent-hover disabled:opacity-40 disabled:hover:bg-accent"
        >
          {loading ? "Running…" : "Run through the pipeline"}
        </button>
        {error && <p className="mt-3 text-[13px] text-warning">{error}</p>}
      </div>

      <div>
        {!result && !loading && (
          <div className="h-full flex items-center justify-center rounded-xl border border-dashed border-border-strong text-[13px] text-text-muted">
            Trace will appear here once you run a message through the pipeline.
          </div>
        )}

        {(result || loading) && (
          <div className="flex flex-col gap-5">
            {loading && !result && (
              <div className="flex items-center gap-2.5 text-[12.5px] font-mono text-text-muted">
                <span className="inline-block h-2 w-2 rounded-full bg-accent animate-pulse" />
                Processing your message…
              </div>
            )}
            <TraceStep visible={visibleStage >= 1} label="1. Classify">
              {result && (
                <div className="flex items-center gap-3 rounded-xl border border-border bg-surface p-4 shadow-[var(--shadow-card)]">
                  <IntentBadge intent={result.classify.intent} />
                  <span className="text-[12px] text-text-muted">confidence</span>
                  <ConfidenceValue value={result.classify.confidence} />
                </div>
              )}
            </TraceStep>

            <TraceStep visible={visibleStage >= 2} label="2. Retrieved precedent">
              {result && (
                <div className="flex flex-col gap-2">
                  {result.draft_reply.grounded_on.length === 0 && (
                    <p className="text-[13px] text-text-muted">No precedent cited.</p>
                  )}
                  {result.draft_reply.grounded_on.map((id, i) => (
                    <div
                      key={id}
                      className="rounded-xl border border-border bg-surface-2 px-4 py-3 shadow-[var(--shadow-card)]"
                    >
                      <p className="text-[11px] font-mono text-text-muted">
                        precedent thread {id} &middot; similarity{" "}
                        {result.draft_reply.retrieval_scores[i]?.toFixed(2) ?? "—"}
                      </p>
                    </div>
                  ))}
                </div>
              )}
            </TraceStep>

            <TraceStep visible={visibleStage >= 3} label="3. Drafted reply">
              {result && (
                <p className="text-[14.5px] leading-[1.65] rounded-xl border border-border bg-surface p-5 shadow-[var(--shadow-card)]">
                  {result.draft_reply.draft}
                </p>
              )}
            </TraceStep>

            <TraceStep visible={visibleStage >= 4} label="4. Decision">
              {result && (
                <div className="rounded-xl border border-border bg-surface p-5 shadow-[var(--shadow-card)]">
                  <EscalationBadge decision={result.decision.decision} />
                  <p className="mt-3 text-[13.5px] leading-[1.65] text-text-secondary">
                    {result.decision.reason}
                  </p>
                </div>
              )}
            </TraceStep>
          </div>
        )}
      </div>
    </div>
  );
}

function TraceStep({
  visible,
  label,
  children,
}: {
  visible: boolean;
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div
      className="transition-all duration-300 ease-out"
      style={{
        opacity: visible ? 1 : 0,
        transform: visible ? "translateY(0)" : "translateY(8px)",
      }}
    >
      <p className="text-[11px] font-mono uppercase tracking-[0.06em] text-text-muted mb-2">{label}</p>
      {visible ? children : <div className="h-14 rounded-xl bg-surface-2 animate-pulse" />}
    </div>
  );
}
