import { useRef, useEffect } from "react";
import ExperimentChart from "./ExperimentChart";

export default function Dashboard({ session, experiments, agentLog, currentExperiment, bestResult, onStop }) {
  const logRef = useRef(null);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [agentLog]);

  const baseline = experiments.length > 0 ? experiments[0].val_bpb : null;
  const improvement = bestResult && baseline
    ? (((baseline - bestResult.val_bpb) / baseline) * 100).toFixed(1)
    : null;

  return (
    <div className="grid grid-cols-[320px_1fr_280px] gap-4 h-[calc(100vh-120px)]">
      {/* Left: Agent Log */}
      <div className="bg-surface-raised border border-border-dim rounded-xl flex flex-col overflow-hidden">
        <div className="px-4 py-3 border-b border-border-dim flex items-center justify-between">
          <span className="text-[10px] font-mono tracking-widest text-text-muted">AGENT LOG</span>
          <span className="text-[10px] font-mono text-text-muted">{agentLog.length}</span>
        </div>
        <div ref={logRef} className="flex-1 overflow-y-auto p-3 space-y-0.5 font-mono text-[11px] leading-relaxed">
          {agentLog.length === 0 && (
            <div className="text-text-muted text-center py-8">Waiting for agent output...</div>
          )}
          {agentLog.map((entry, i) => (
            <div key={i} className={`log-entry px-2 py-0.5 rounded ${
              entry.level === "result" ? "text-accent bg-accent-dim" :
              entry.level === "error" ? "text-danger bg-danger-dim" :
              entry.level === "code" ? "text-text-secondary bg-surface-overlay" :
              "text-text-muted"
            }`}>
              <span className="text-text-muted/50 mr-2 select-none">
                {entry.level === "result" ? "\u25B6" : entry.level === "error" ? "\u2718" : entry.level === "code" ? "\u276F" : "\u00B7"}
              </span>
              {entry.text}
            </div>
          ))}
        </div>
      </div>

      {/* Center: Chart */}
      <div className="bg-surface-raised border border-border-dim rounded-xl flex flex-col overflow-hidden">
        <div className="px-4 py-3 border-b border-border-dim flex items-center justify-between">
          <span className="text-[10px] font-mono tracking-widest text-text-muted">EXPERIMENTS</span>
          <div className="flex gap-3 text-[10px] font-mono">
            <span className="text-accent flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-accent inline-block" /> keep
            </span>
            <span className="text-discard flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-discard inline-block" /> discard
            </span>
            <span className="text-danger flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-danger inline-block" /> crash
            </span>
          </div>
        </div>
        <div className="flex-1 p-4">
          <ExperimentChart experiments={experiments} />
        </div>
      </div>

      {/* Right: Stats */}
      <div className="space-y-4 flex flex-col">
        {/* Session Status */}
        <div className="bg-surface-raised border border-border-dim rounded-xl p-4 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-mono tracking-widest text-text-muted">SESSION</span>
            {session.active ? (
              <span className="text-[10px] font-mono text-accent flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-accent pulse-dot" /> ACTIVE
              </span>
            ) : (
              <span className="text-[10px] font-mono text-text-muted">IDLE</span>
            )}
          </div>
          {session.branch && (
            <div className="font-mono text-[11px] text-text-secondary truncate">{session.branch}</div>
          )}
          {session.active && (
            <button
              onClick={onStop}
              className="w-full mt-2 py-2 rounded-lg border border-danger/30 bg-danger-dim text-danger text-[11px] font-mono tracking-wider hover:bg-danger/20 transition-colors"
            >
              STOP SESSION
            </button>
          )}
        </div>

        {/* Best Result */}
        <div className="bg-surface-raised border border-border-dim rounded-xl p-4 space-y-1 glow-green">
          <span className="text-[10px] font-mono tracking-widest text-text-muted">BEST RESULT</span>
          {bestResult ? (
            <>
              <div className="text-3xl font-mono text-accent font-bold tabular-nums">
                {bestResult.val_bpb.toFixed(4)}
              </div>
              <div className="text-[11px] text-text-secondary">val_bpb</div>
              {improvement && (
                <div className="text-[11px] text-accent font-mono">{"\u2193"} {improvement}% from baseline</div>
              )}
              <div className="text-[10px] text-text-muted font-mono mt-1 truncate">
                {bestResult.commit?.slice(0, 7)} {bestResult.description}
              </div>
            </>
          ) : (
            <div className="text-2xl font-mono text-text-muted">--</div>
          )}
        </div>

        {/* Current Experiment */}
        <div className="bg-surface-raised border border-border-dim rounded-xl p-4 space-y-1 flex-1">
          <span className="text-[10px] font-mono tracking-widest text-text-muted">CURRENT</span>
          {currentExperiment ? (
            <>
              <div className="text-lg font-mono text-text-primary">#{currentExperiment.n}</div>
              {currentExperiment.ticks.length > 0 && (() => {
                const last = currentExperiment.ticks[currentExperiment.ticks.length - 1];
                const remaining = Math.max(0, 300 - last.elapsed);
                const mins = Math.floor(remaining / 60);
                const secs = Math.floor(remaining % 60);
                return (
                  <div className="space-y-1 mt-2">
                    <div className="flex justify-between text-[11px] font-mono">
                      <span className="text-text-muted">loss</span>
                      <span className="text-text-primary">{last.loss.toFixed(4)}</span>
                    </div>
                    <div className="flex justify-between text-[11px] font-mono">
                      <span className="text-text-muted">step</span>
                      <span className="text-text-primary">{last.step}</span>
                    </div>
                    <div className="flex justify-between text-[11px] font-mono">
                      <span className="text-text-muted">remaining</span>
                      <span className="text-accent">{mins}:{secs.toString().padStart(2, "0")}</span>
                    </div>
                  </div>
                );
              })()}
            </>
          ) : (
            <div className="text-sm text-text-muted font-mono">
              {experiments.length > 0 ? "Between experiments" : "Waiting..."}
            </div>
          )}
        </div>

        {/* Stats */}
        <div className="bg-surface-raised border border-border-dim rounded-xl p-4">
          <div className="grid grid-cols-2 gap-3">
            <div>
              <div className="text-[10px] font-mono text-text-muted">TOTAL</div>
              <div className="text-lg font-mono text-text-primary">{experiments.length}</div>
            </div>
            <div>
              <div className="text-[10px] font-mono text-text-muted">KEPT</div>
              <div className="text-lg font-mono text-accent">
                {experiments.filter((e) => e.status === "keep").length}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
