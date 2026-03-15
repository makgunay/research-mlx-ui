import { useState } from "react";

function DiffLine({ line }) {
  if (line.startsWith("+") && !line.startsWith("+++")) {
    return <div className="text-accent bg-accent-dim px-2">{line}</div>;
  }
  if (line.startsWith("-") && !line.startsWith("---")) {
    return <div className="text-danger bg-danger-dim px-2">{line}</div>;
  }
  if (line.startsWith("@@")) {
    return <div className="text-blue-400/70 px-2">{line}</div>;
  }
  if (line.startsWith("diff ") || line.startsWith("index ") || line.startsWith("---") || line.startsWith("+++")) {
    return <div className="text-text-muted/50 px-2">{line}</div>;
  }
  return <div className="text-text-secondary/70 px-2">{line}</div>;
}

function SyntaxDiff({ diff }) {
  if (!diff || diff === "No diff available") {
    return <div className="text-text-muted px-2">No diff available</div>;
  }
  const lines = diff.split("\n");
  return (
    <div className="font-mono text-[11px] leading-relaxed overflow-x-auto whitespace-pre">
      {lines.map((line, i) => (
        <DiffLine key={i} line={line} />
      ))}
    </div>
  );
}

export default function History({ experiments }) {
  const [expandedCommit, setExpandedCommit] = useState(null);
  const [diffData, setDiffData] = useState({});

  const toggleDiff = async (commit) => {
    if (!commit || commit === "-" || commit === "baseline") return;
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

      {[...experiments].reverse().map((exp) => (
        <div key={`${exp.n}-${exp.val_bpb}`} className="bg-surface-raised border border-border-dim rounded-xl overflow-hidden">
          <button
            type="button"
            onClick={() => toggleDiff(exp.commit)}
            className={`w-full px-5 py-3 flex items-center gap-4 text-left hover:bg-surface-overlay/50 transition-colors ${
              exp.commit === "-" || exp.commit === "baseline" ? "cursor-default" : ""
            }`}
          >
            <span className="font-mono text-xs text-text-muted w-8">#{exp.n}</span>
            <span className={`w-2 h-2 rounded-full flex-shrink-0 ${
              exp.status === "keep" ? "bg-accent" : exp.status === "crash" ? "bg-danger" : "bg-discard"
            }`} />
            <span className={`font-mono text-sm tabular-nums w-20 ${
              exp.status === "keep" ? "text-accent" : "text-text-primary"
            }`}>
              {typeof exp.val_bpb === "number" ? exp.val_bpb.toFixed(4) : "\u2014"}
            </span>
            <span className={`text-[11px] px-1.5 py-0.5 rounded font-mono ${
              exp.status === "keep" ? "bg-accent-dim text-accent" :
              exp.status === "crash" ? "bg-danger-dim text-danger" :
              "bg-surface-overlay text-discard"
            }`}>
              {exp.status}
            </span>
            <span className="text-sm text-text-secondary flex-1 truncate">{exp.description}</span>
            {exp.commit && exp.commit !== "-" && exp.commit !== "baseline" && (
              <>
                <span className="font-mono text-[10px] text-text-muted">{exp.commit.slice(0, 7)}</span>
                <span className="text-text-muted text-xs">{expandedCommit === exp.commit ? "\u25B2" : "\u25BC"}</span>
              </>
            )}
          </button>

          {expandedCommit === exp.commit && (
            <div className="border-t border-border-dim bg-surface-overlay p-4">
              <SyntaxDiff diff={diffData[exp.commit]} />
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
