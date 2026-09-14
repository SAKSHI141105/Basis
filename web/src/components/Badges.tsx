export function IntentBadge({ intent }: { intent: string }) {
  return (
    <span className="inline-flex items-center rounded-full border border-border bg-surface-2 px-2.5 py-1 text-[12px] font-mono text-text-secondary">
      {intent}
    </span>
  );
}

export function EscalationBadge({ decision }: { decision: string }) {
  const isAuto = decision === "auto_handle";
  return (
    <span
      className={
        "inline-flex items-center rounded-full border px-2.5 py-1 text-[12px] font-mono " +
        (isAuto
          ? "border-success-border bg-success-bg text-success"
          : "border-warning-border bg-warning-bg text-warning")
      }
    >
      {decision}
    </span>
  );
}

export function ConfidenceValue({ value }: { value: number }) {
  return <span className="font-mono text-[13px] tabular-nums text-text-secondary">{value.toFixed(2)}</span>;
}
