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
    <div className="mx-auto max-w-[1120px] px-6 py-12 grid grid-cols-1 md:grid-cols-[360px_1fr] gap-10">
      <div>
        <h1 className="text-[18px] font-medium mb-1">Live demo</h1>
        <p className="text-[13px] text-text-secondary mb-6">
          Pick a real customer message from the golden set, or write your own.
        </p>

        <div className="flex flex-col gap-2 mb-6">
          {samples.map((s) => (
            <button
              key={s.thread_id}
              onClick={() => {
                setSelectedId(s.thread_id);
                setMessage(s.message);
              }}
              className={
                "text-left rounded-md border px-3 py-2.5 text-[13px] leading-[1.4] transition-colors " +
                (selectedId === s.thread_id
                  ? "border-accent-border bg-accent-bg text-text"
                  : "border-border bg-surface text-text-secondary hover:border-border-strong")
              }
            >
              {s.message.length > 90 ? s.message.slice(0, 90) + "…" : s.message}
            </button>
          ))}
          {samples.length === 0 && (
            <p className="text-[12px] font-mono text-text-muted">
              No samples loaded — is the API running at the configured URL?
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
          className="w-full rounded-md border border-border bg-surface px-3 py-2.5 text-[13.5px] leading-[1.5] text-text placeholder:text-text-muted resize-none"
        />
        <button
          onClick={handleRun}
          disabled={loading || !message.trim()}
          className="mt-3 w-full rounded-md bg-accent px-4 py-2.5 text-[13px] font-medium text-white disabled:opacity-40"
        >
          {loading ? "Running…" : "Run through the pipeline"}
        </button>
        {error && <p className="mt-3 text-[13px] text-warning">{error}</p>}
      </div>

      <div>
        {!result && !loading && (
          <div className="h-full flex items-center justify-center rounded-lg border border-dashed border-border text-[13px] text-text-muted">
            Trace will appear here once you run a message through the pipeline.
          </div>
        )}

        {(result || loading) && (
          <div className="flex flex-col gap-4">
            <TraceStep visible={visibleStage >= 1} label="1. Classify">
              {result && (
                <div className="flex items-center gap-3">
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
                    <div key={id} className="rounded-md border border-border bg-surface-2 px-3 py-2">
                      <p className="text-[11px] font-mono text-text-muted mb-1">
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
                <p className="text-[14px] leading-[1.6] rounded-md border border-border bg-surface p-4">
                  {result.draft_reply.draft}
                </p>
              )}
            </TraceStep>

            <TraceStep visible={visibleStage >= 4} label="4. Decision">
              {result && (
                <div className="rounded-md border border-border bg-surface p-4">
                  <EscalationBadge decision={result.decision.decision} />
                  <p className="mt-3 text-[13.5px] leading-[1.6] text-text-secondary">
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
      {visible ? children : <div className="h-10 rounded-md bg-surface-2 animate-pulse" />}
    </div>
  );
}
