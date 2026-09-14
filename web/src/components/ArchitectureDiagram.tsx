const box = {
  fill: "var(--surface)",
  stroke: "var(--border-strong)",
};

const labelStyle = { fill: "var(--text)", fontFamily: "var(--font-mono)" };
const subLabelStyle = { fill: "var(--text-muted)", fontFamily: "var(--font-mono)" };
const lineStyle = { stroke: "var(--border-strong)" };

export function ArchitectureDiagram() {
  return (
    <svg
      viewBox="0 0 900 420"
      role="img"
      aria-label="System architecture: offline pipeline feeds online serving, which is used by both the evaluation harness and the demo frontend"
      className="w-full h-auto"
    >
      <title>System architecture</title>

      {/* Offline pipeline */}
      <rect x="40" y="20" width="820" height="120" rx="8" style={box} strokeWidth="1.5" />
      <text x="60" y="46" fontSize="11" style={subLabelStyle} letterSpacing="0.06em">
        OFFLINE PIPELINE (run once, artifacts cached)
      </text>
      {[
        "1. ingest + reconstruct",
        "2. clean",
        "3. embed + cluster → taxonomy",
        "4. train / eval split",
        "5. build retrieval index",
        "6. train baselines",
      ].map((step, i) => (
        <text key={step} x={60 + (i % 3) * 265} y={72 + Math.floor(i / 3) * 26} fontSize="12.5" style={labelStyle}>
          {step}
        </text>
      ))}

      {/* arrow down */}
      <line x1="450" y1="140" x2="450" y2="176" style={lineStyle} strokeWidth="1.5" markerEnd="url(#arrow)" />
      <text x="466" y="162" fontSize="10.5" style={subLabelStyle}>
        taxonomy.yaml, index, baseline model
      </text>

      {/* Online serving */}
      <rect x="220" y="180" width="460" height="110" rx="8" style={box} strokeWidth="1.5" />
      <text x="240" y="206" fontSize="11" style={subLabelStyle} letterSpacing="0.06em">
        ONLINE SERVING (FastAPI)
      </text>
      {["POST /classify", "POST /draft-reply", "POST /decide", "POST /pipeline (chains all three)"].map(
        (ep, i) => (
          <text key={ep} x={240 + (i % 2) * 220} y={234 + Math.floor(i / 2) * 26} fontSize="12.5" style={labelStyle}>
            {ep}
          </text>
        )
      )}

      {/* branch lines */}
      <line x1="330" y1="290" x2="200" y2="330" style={lineStyle} strokeWidth="1.5" markerEnd="url(#arrow)" />
      <line x1="570" y1="290" x2="700" y2="330" style={lineStyle} strokeWidth="1.5" markerEnd="url(#arrow)" />

      {/* Eval harness */}
      <rect x="40" y="336" width="340" height="70" rx="8" style={box} strokeWidth="1.5" />
      <text x="60" y="362" fontSize="11" style={subLabelStyle} letterSpacing="0.06em">
        EVALUATION HARNESS
      </text>
      <text x="60" y="386" fontSize="12.5" style={labelStyle}>
        golden_set.jsonl → eval_report.json / .md
      </text>

      {/* Demo frontend */}
      <rect x="500" y="336" width="360" height="70" rx="8" style={box} strokeWidth="1.5" />
      <text x="520" y="362" fontSize="11" style={subLabelStyle} letterSpacing="0.06em">
        DEMO FRONTEND (this app)
      </text>
      <text x="520" y="386" fontSize="12.5" style={labelStyle}>
        live demo panel + eval dashboard
      </text>

      <defs>
        <marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" style={{ fill: "var(--border-strong)" }} />
        </marker>
      </defs>
    </svg>
  );
}
