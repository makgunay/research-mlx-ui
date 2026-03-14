import { useState, useEffect } from "react";

export default function Results({ experiments }) {
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (experiments.length > 0 && !summary) {
      setLoading(true);
      fetch("/api/results/summary")
        .then((r) => r.json())
        .then(setSummary)
        .catch(() => setSummary({ summary: "Failed to generate summary.", insights: [] }))
        .finally(() => setLoading(false));
    }
  }, [experiments.length]);

  const kept = experiments.filter((e) => e.status === "keep");
  const discarded = experiments.filter((e) => e.status === "discard");
  const best = kept.length ? kept.reduce((a, b) => (a.val_bpb < b.val_bpb ? a : b)) : null;
  const baseline = experiments.length > 0 ? experiments[0] : null;

  return (
    <div className="max-w-2xl mx-auto space-y-6 pt-6">
      <h2 className="font-mono text-sm tracking-widest text-text-muted text-center">SESSION RESULTS</h2>

      {/* Stats Grid */}
      <div className="grid grid-cols-4 gap-3">
        {[
          { label: "TOTAL", value: experiments.length, color: "text-text-primary" },
          { label: "KEPT", value: kept.length, color: "text-accent" },
          { label: "DISCARDED", value: discarded.length, color: "text-discard" },
          { label: "BEST BPB", value: best ? best.val_bpb.toFixed(4) : "--", color: "text-accent" },
        ].map(({ label, value, color }) => (
          <div key={label} className="bg-surface-raised border border-border-dim rounded-xl p-4 text-center">
            <div className="text-[10px] font-mono text-text-muted tracking-widest mb-1">{label}</div>
            <div className={`text-xl font-mono font-bold tabular-nums ${color}`}>{value}</div>
          </div>
        ))}
      </div>

      {/* Improvement */}
      {best && baseline && (
        <div className="bg-surface-raised border border-accent/20 rounded-xl p-5 text-center glow-green">
          <div className="text-[10px] font-mono text-text-muted tracking-widest mb-2">IMPROVEMENT FROM BASELINE</div>
          <div className="text-4xl font-mono text-accent font-bold">
            {"\u2193"} {(((baseline.val_bpb - best.val_bpb) / baseline.val_bpb) * 100).toFixed(1)}%
          </div>
          <div className="text-sm text-text-secondary mt-2 font-mono">
            {baseline.val_bpb.toFixed(4)} {"\u2192"} {best.val_bpb.toFixed(4)}
          </div>
        </div>
      )}

      {/* AI Summary */}
      <div className="bg-surface-raised border border-border-dim rounded-xl p-5 space-y-3">
        <span className="text-[10px] font-mono tracking-widest text-text-muted">AI SUMMARY</span>
        {loading ? (
          <div className="text-sm text-text-muted animate-pulse">Generating summary...</div>
        ) : summary ? (
          <>
            <p className="text-sm text-text-secondary leading-relaxed">{summary.summary}</p>
            {summary.insights?.length > 0 && (
              <ul className="space-y-1.5 mt-3">
                {summary.insights.map((insight, i) => (
                  <li key={i} className="text-sm text-text-secondary flex gap-2">
                    <span className="text-accent flex-shrink-0">{"\u2022"}</span>
                    {insight}
                  </li>
                ))}
              </ul>
            )}
          </>
        ) : (
          <p className="text-sm text-text-muted">Run experiments to see a summary.</p>
        )}
      </div>
    </div>
  );
}
