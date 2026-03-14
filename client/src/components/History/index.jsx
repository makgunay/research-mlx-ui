import { useState } from "react";

export default function History({ experiments }) {
  const [expandedCommit, setExpandedCommit] = useState(null);
  const [diffData, setDiffData] = useState({});

  const toggleDiff = async (commit) => {
    if (expandedCommit === commit) {
      setExpandedCommit(null);
      return;
    }
    setExpandedCommit(commit);
    if (!diffData[commit]) {
      try {
        const res = await fetch(`/api/experiments/${commit}`);
        const data = await res.json();
        setDiffData((d) => ({ ...d, [commit]: data?.diff || "No diff available" }));
      } catch {
        setDiffData((d) => ({ ...d, [commit]: "Failed to load diff" }));
      }
    }
  };

  if (experiments.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-text-muted font-mono text-sm">
        No experiments recorded yet. Start a session first.
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto space-y-2">
      <div className="flex items-center justify-between mb-6">
        <h2 className="font-mono text-sm tracking-widest text-text-muted">EXPERIMENT HISTORY</h2>
        <span className="font-mono text-[11px] text-text-muted">
          {experiments.length} experiments / {experiments.filter((e) => e.status === "keep").length} kept
        </span>
      </div>

      {experiments.map((exp) => (
        <div key={exp.commit || exp.n} className="bg-surface-raised border border-border-dim rounded-xl overflow-hidden">
          <button
            onClick={() => exp.commit && toggleDiff(exp.commit)}
            className="w-full px-5 py-3 flex items-center gap-4 text-left hover:bg-surface-overlay/50 transition-colors"
          >
            <span className="font-mono text-xs text-text-muted w-8">#{exp.n}</span>
            <span className={`w-2 h-2 rounded-full flex-shrink-0 ${
              exp.status === "keep" ? "bg-accent" : exp.status === "crash" ? "bg-danger" : "bg-discard"
            }`} />
            <span className="font-mono text-sm text-text-primary tabular-nums w-20">
              {exp.val_bpb.toFixed(4)}
            </span>
            <span className="text-sm text-text-secondary flex-1 truncate">{exp.description}</span>
            <span className="font-mono text-[10px] text-text-muted">{exp.commit?.slice(0, 7)}</span>
            <span className="text-text-muted text-xs">{expandedCommit === exp.commit ? "\u25B2" : "\u25BC"}</span>
          </button>

          {expandedCommit === exp.commit && (
            <div className="border-t border-border-dim bg-surface-overlay p-4">
              <pre className="font-mono text-[11px] leading-relaxed text-text-secondary overflow-x-auto whitespace-pre">
                {diffData[exp.commit] || "Loading..."}
              </pre>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
