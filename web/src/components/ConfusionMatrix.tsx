export function ConfusionMatrix({ labels, matrix }: { labels: string[]; matrix: number[][] }) {
  const max = Math.max(1, ...matrix.flat());

  return (
    <div className="overflow-x-auto">
      <table className="border-collapse">
        <thead>
          <tr>
            <th className="p-1" />
            <th colSpan={labels.length} className="text-center text-[10px] font-mono text-text-muted pb-1">
              predicted
            </th>
          </tr>
          <tr>
            <th className="p-1" />
            {labels.map((l) => (
              <th key={l} className="p-1 text-[10px] font-mono text-text-muted font-normal align-bottom">
                <span className="block w-12 truncate" title={l}>
                  {l.slice(0, 4)}
                </span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={labels[i]}>
              {i === 0 && (
                <th
                  rowSpan={matrix.length}
                  className="p-1 text-[10px] font-mono text-text-muted align-middle"
                  style={{ writingMode: "vertical-rl" }}
                >
                  actual
                </th>
              )}
              <th className="p-1 pr-2 text-[10px] font-mono text-text-muted font-normal text-right">
                <span className="block max-w-16 truncate" title={labels[i]}>
                  {labels[i]}
                </span>
              </th>
              {row.map((value, j) => {
                const intensity = value / max;
                const isDiagonal = i === j;
                return (
                  <td
                    key={j}
                    className="w-12 h-9 text-center text-[11px] font-mono tabular-nums border border-border"
                    style={{
                      background: isDiagonal
                        ? `color-mix(in srgb, var(--success-bg) ${20 + intensity * 80}%, var(--surface))`
                        : `color-mix(in srgb, var(--warning-bg) ${intensity * 90}%, var(--surface))`,
                      color: intensity > 0.5 ? "var(--text)" : "var(--text-secondary)",
                    }}
                  >
                    {value || ""}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
