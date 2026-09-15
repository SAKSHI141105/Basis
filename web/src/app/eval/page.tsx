import { getEvalReport } from "@/lib/api";
import { ConfusionMatrix } from "@/components/ConfusionMatrix";
import { EscalationBadge, IntentBadge } from "@/components/Badges";

const SYSTEM_LABELS: Record<string, string> = {
  trivial: "Trivial baseline",
  simple: "Simple baseline (TF-IDF + LogReg)",
  main: "Main system",
};

export default async function EvalDashboardPage() {
  const report = await getEvalReport().catch(() => null);

  if (!report) {
    return (
      <div className="mx-auto max-w-[1120px] px-6 py-16">
        <h1 className="font-display text-[24px] font-medium mb-4 text-text">Eval dashboard</h1>
        <div className="rounded-xl border border-border bg-surface-2 p-7 text-[13.5px] leading-[1.65] text-text-secondary font-mono">
          <code>eval_report.json</code> not yet generated. Run{" "}
          <code>make eval-fast</code> (replays the committed cache) or{" "}
          <code>make eval-live</code> (calls the real API) from the project root,
          then reload this page.
        </div>
      </div>
    );
  }

  const skipped = report.skipped_example_ids ?? [];

  return (
    <div className="mx-auto max-w-[1120px] px-6 py-16">
      <h1 className="font-display text-[24px] font-medium mb-2 text-text">Eval dashboard</h1>
      <p className="text-[13.5px] text-text-secondary mb-2">
        Golden set: {report.golden_set_size} examples
        {skipped.length > 0 && (
          <>
            {" "}
            &middot; {skipped.length} skipped (call failure) &mdash; main system reflects{" "}
            {report.main_system_examples_evaluated ?? "?"} examples
          </>
        )}
      </p>

      {/* Misleading-number note gets equal visual weight, per Design Brief 3.3 */}
      <section className="mt-9 rounded-xl border border-warning-border bg-warning-bg p-7 shadow-[var(--shadow-card)]">
        <h2 className="text-[12px] font-mono uppercase tracking-[0.06em] text-warning mb-2.5">
          What&apos;s misleading about the headline number
        </h2>
        <p className="text-[14.5px] leading-[1.65] text-text">{report.misleading_number_note}</p>
      </section>

      <section className="mt-12">
        <h2 className="text-[12.5px] font-medium uppercase tracking-[0.08em] text-text-muted mb-4">
          Intent classification
        </h2>
        <MetricsTable
          columns={["Accuracy", "Macro-F1"]}
          rows={Object.entries(report.intent_metrics).map(([system, m]) => [
            SYSTEM_LABELS[system] ?? system,
            m.accuracy.toFixed(3),
            m.macro_f1.toFixed(3),
          ])}
        />
      </section>

      <section className="mt-12">
        <h2 className="text-[12.5px] font-medium uppercase tracking-[0.08em] text-text-muted mb-4">
          Per-intent F1 &mdash; main system
        </h2>
        <p className="text-[12.5px] text-text-muted mb-4">
          One aggregate macro-F1 hides which intents the classifier actually
          struggles with (TRD 7.2) &mdash; broken out below, worst first.
        </p>
        <div className="overflow-x-auto rounded-xl border border-border shadow-[var(--shadow-card)]">
          <table className="w-full border-collapse text-[13px]">
            <thead>
              <tr className="border-b border-border bg-surface-2">
                <th className="text-left font-medium px-5 py-3">Intent</th>
                <th className="text-right font-medium px-5 py-3 font-mono">F1</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(report.intent_metrics.main.per_intent_f1)
                .sort((a, b) => a[1] - b[1])
                .map(([intent, f1]) => (
                  <tr key={intent} className="border-b border-border last:border-0 bg-surface">
                    <td className="px-5 py-3">
                      <IntentBadge intent={intent} />
                    </td>
                    <td className="px-5 py-3 text-right font-mono tabular-nums">{f1.toFixed(3)}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="mt-12">
        <h2 className="text-[12.5px] font-medium uppercase tracking-[0.08em] text-text-muted mb-4">
          Escalation decision
        </h2>
        <MetricsTable
          columns={["Precision", "Recall", "F1", "Cost-weighted"]}
          rows={Object.entries(report.escalation_metrics).map(([system, m]) => [
            SYSTEM_LABELS[system] ?? system,
            m.precision.toFixed(3),
            m.recall.toFixed(3),
            m.f1.toFixed(3),
            m.cost_weighted_score.toFixed(3),
          ])}
        />
      </section>

      <section className="mt-12">
        <h2 className="text-[12.5px] font-medium uppercase tracking-[0.08em] text-text-muted mb-4">
          Confusion matrix &mdash; main system
        </h2>
        <div className="rounded-xl border border-border bg-surface p-5 shadow-[var(--shadow-card)]">
          <ConfusionMatrix
            labels={report.intent_metrics.main.labels}
            matrix={report.intent_metrics.main.confusion_matrix}
          />
        </div>
      </section>

      {Object.keys(report.judge_summary).length > 0 && (
        <section className="mt-12">
          <h2 className="text-[12.5px] font-medium uppercase tracking-[0.08em] text-text-muted mb-4">
            LLM judge &mdash; reply quality (1-5)
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            {Object.entries(report.judge_summary).map(([k, v]) => (
              <div
                key={k}
                className="rounded-xl border border-border bg-surface p-5 shadow-[var(--shadow-card)]"
              >
                <p className="text-[11px] font-mono text-text-muted mb-1.5">{k.replace("mean_", "")}</p>
                <p className="font-display text-[24px] font-medium tabular-nums text-text">{v.toFixed(2)}</p>
              </div>
            ))}
          </div>
        </section>
      )}

      <section className="mt-12">
        <h2 className="text-[12.5px] font-medium uppercase tracking-[0.08em] text-text-muted mb-4">
          Human-agreement study
        </h2>
        {report.human_agreement ? (
          <div className="rounded-xl border border-border bg-surface p-7 text-[13.5px] leading-[1.75] font-mono shadow-[var(--shadow-card)]">
            <p>n = {report.human_agreement.n}</p>
            <p>Cohen&apos;s kappa: {report.human_agreement.cohen_kappa.toFixed(3)}</p>
            <p>
              Spearman r: {report.human_agreement.spearman_r.toFixed(3)} (p=
              {report.human_agreement.spearman_p.toFixed(4)})
            </p>
            <p>Mean absolute diff: {report.human_agreement.mean_absolute_diff.toFixed(3)}</p>
          </div>
        ) : (
          <div className="rounded-xl border border-border bg-surface-2 p-7 text-[13.5px] leading-[1.65] text-text-secondary">
            Not performed &mdash; no independent human rater was available for this
            build. Using another LLM as a stand-in would produce circular, not
            weaker, evidence, so this is marked not-performed rather than faked.
            See <code className="font-mono text-[12.5px]">REPORT.md</code> §5 and{" "}
            <code className="font-mono text-[12.5px]">DECISION_LOG.md</code>.
          </div>
        )}
      </section>

      {report.failure_examples.length > 0 && (
        <section className="mt-12 mb-20">
          <h2 className="text-[12.5px] font-medium uppercase tracking-[0.08em] text-text-muted mb-4">
            Failure examples
          </h2>
          <div className="flex flex-col gap-3">
            {report.failure_examples.map((f) => (
              <div
                key={f.thread_id}
                className="rounded-xl border border-border bg-surface p-6 shadow-[var(--shadow-card)]"
              >
                <p className="font-display text-[15px] italic leading-[1.6] mb-3.5 text-text">
                  &ldquo;{f.customer_msg}&rdquo;
                </p>
                <div className="flex flex-wrap gap-x-6 gap-y-2 text-[12px]">
                  <span className="text-text-muted">
                    intent: true <IntentBadge intent={f.true_intent} /> pred{" "}
                    <IntentBadge intent={f.pred_intent} />
                  </span>
                  <span className="text-text-muted">
                    escalation: true <EscalationBadge decision={f.true_escalation} /> pred{" "}
                    <EscalationBadge decision={f.pred_escalation} />
                  </span>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

function MetricsTable({ columns, rows }: { columns: string[]; rows: (string | number)[][] }) {
  return (
    <div className="overflow-x-auto rounded-xl border border-border shadow-[var(--shadow-card)]">
      <table className="w-full border-collapse text-[13px]">
        <thead>
          <tr className="border-b border-border bg-surface-2">
            <th className="text-left font-medium px-5 py-3">System</th>
            {columns.map((c) => (
              <th key={c} className="text-right font-medium px-5 py-3 font-mono">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className="border-b border-border last:border-0 bg-surface">
              <td className="px-5 py-3">{row[0]}</td>
              {row.slice(1).map((v, j) => (
                <td key={j} className="px-5 py-3 text-right font-mono tabular-nums">
                  {v}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
