import { ScatterChart, Scatter, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";

const STATUS_COLORS = {
  keep: "#22c55e",
  discard: "#64748b",
  crash: "#ef4444",
};

function CustomTooltip({ active, payload }) {
  if (!active || !payload?.[0]) return null;
  const d = payload[0].payload;
  return (
    <div className="bg-surface-overlay border border-border rounded-lg px-3 py-2 font-mono text-[11px] shadow-xl">
      <div className="text-text-primary">#{d.n} &mdash; {d.val_bpb.toFixed(4)} bpb</div>
      <div className="text-text-muted mt-0.5">{d.description}</div>
      <div className={`mt-1 ${d.status === "keep" ? "text-accent" : d.status === "crash" ? "text-danger" : "text-discard"}`}>
        {d.status}
      </div>
    </div>
  );
}

export default function ExperimentChart({ experiments }) {
  if (experiments.length === 0) {
    return (
      <div className="h-full flex items-center justify-center text-text-muted font-mono text-sm">
        No experiments yet
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
        <XAxis
          dataKey="n"
          type="number"
          domain={[0, "auto"]}
          tick={{ fill: "#4a5a6e", fontSize: 10, fontFamily: "var(--font-mono)" }}
          axisLine={{ stroke: "#1a2332" }}
          tickLine={{ stroke: "#1a2332" }}
          label={{ value: "experiment #", position: "bottom", offset: 0, fill: "#4a5a6e", fontSize: 10, fontFamily: "var(--font-mono)" }}
        />
        <YAxis
          dataKey="val_bpb"
          type="number"
          domain={["auto", "auto"]}
          tick={{ fill: "#4a5a6e", fontSize: 10, fontFamily: "var(--font-mono)" }}
          axisLine={{ stroke: "#1a2332" }}
          tickLine={{ stroke: "#1a2332" }}
          label={{ value: "val_bpb (lower = better)", angle: -90, position: "insideLeft", offset: 10, fill: "#4a5a6e", fontSize: 10, fontFamily: "var(--font-mono)" }}
        />
        <Tooltip content={<CustomTooltip />} cursor={false} />
        <Scatter data={experiments} shape="circle">
          {experiments.map((entry, i) => (
            <Cell
              key={i}
              fill={STATUS_COLORS[entry.status] || "#64748b"}
              fillOpacity={entry.status === "discard" ? 0.4 : 0.9}
              r={entry.status === "keep" ? 6 : 4}
            />
          ))}
        </Scatter>
      </ScatterChart>
    </ResponsiveContainer>
  );
}
