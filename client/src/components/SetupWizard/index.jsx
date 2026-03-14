import { useState } from "react";

const FOCUS_OPTIONS = [
  { key: "speed", label: "Speed", icon: "\u26A1", desc: "Maximize throughput" },
  { key: "memory", label: "Memory", icon: "\u2B1B", desc: "Fit larger models" },
  { key: "accuracy", label: "Accuracy", icon: "\u25CE", desc: "Push BPB lower" },
  { key: "optimizer", label: "Optimizer", icon: "\u2699\uFE0F", desc: "Tune Muon/AdamW" },
];

export default function SetupWizard({ hardware, onStart }) {
  const [focus, setFocus] = useState([]);
  const [hints, setHints] = useState("");
  const [starting, setStarting] = useState(false);

  const toggleFocus = (key) =>
    setFocus((f) => (f.includes(key) ? f.filter((k) => k !== key) : [...f, key]));

  const handleStart = async () => {
    setStarting(true);
    try {
      await onStart({
        focusAreas: focus,
        hints,
        branchName: new Date().toISOString().slice(0, 10),
      });
    } catch {
      setStarting(false);
    }
  };

  return (
    <div className="max-w-xl mx-auto space-y-6 pt-12">
      {/* Title */}
      <div className="text-center space-y-2 mb-10">
        <h1 className="text-2xl font-mono tracking-tight text-text-primary">
          New Research Session
        </h1>
        <p className="text-sm text-text-muted">
          Configure and launch autonomous ML experiments on your Mac
        </p>
      </div>

      {/* Hardware Card */}
      <div className="bg-surface-raised border border-border-dim rounded-xl p-5 space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-[10px] font-mono tracking-widest text-text-muted">HARDWARE</span>
          <span className="text-[10px] font-mono text-accent">DETECTED</span>
        </div>
        {hardware ? (
          <div className="space-y-1">
            <div className="text-lg font-mono text-text-primary">{hardware.chip}</div>
            <div className="text-sm text-text-secondary">{hardware.memory} Unified Memory</div>
            <div className="mt-3 flex gap-3 text-[10px] font-mono text-text-muted">
              <span>depth={hardware.recommendations.depth}</span>
              <span>batch={hardware.recommendations.device_batch_size}</span>
              <span>seq={hardware.recommendations.max_seq_len}</span>
            </div>
          </div>
        ) : (
          <div className="text-sm text-text-muted animate-pulse">Detecting hardware...</div>
        )}
      </div>

      {/* Focus Selector */}
      <div className="bg-surface-raised border border-border-dim rounded-xl p-5 space-y-3">
        <span className="text-[10px] font-mono tracking-widest text-text-muted">RESEARCH FOCUS</span>
        <div className="grid grid-cols-2 gap-2">
          {FOCUS_OPTIONS.map(({ key, label, icon, desc }) => (
            <button
              key={key}
              onClick={() => toggleFocus(key)}
              className={`text-left p-3 rounded-lg border transition-all ${
                focus.includes(key)
                  ? "border-accent bg-accent-dim text-text-primary"
                  : "border-border-dim bg-surface-overlay text-text-secondary hover:border-border"
              }`}
            >
              <div className="text-lg mb-1">{icon}</div>
              <div className="text-sm font-medium">{label}</div>
              <div className="text-[11px] text-text-muted mt-0.5">{desc}</div>
            </button>
          ))}
        </div>
        <p className="text-[11px] text-text-muted">
          Select none for open exploration. Multi-select OK.
        </p>
      </div>

      {/* Hints */}
      <div className="bg-surface-raised border border-border-dim rounded-xl p-5 space-y-3">
        <span className="text-[10px] font-mono tracking-widest text-text-muted">HINTS (OPTIONAL)</span>
        <textarea
          value={hints}
          onChange={(e) => setHints(e.target.value)}
          placeholder="Any specific ideas to try? e.g. 'try SwiGLU activation' or 'explore learning rate warmup'"
          className="w-full h-24 bg-surface-overlay border border-border-dim rounded-lg p-3 text-sm text-text-primary placeholder-text-muted resize-none focus:outline-none focus:border-accent/50 transition-colors"
        />
      </div>

      {/* Start Button */}
      <button
        onClick={handleStart}
        disabled={!hardware || starting}
        className="w-full py-3.5 rounded-xl font-mono text-sm tracking-wider bg-accent text-surface font-semibold hover:bg-accent/90 transition-all disabled:opacity-40 disabled:cursor-not-allowed glow-green"
      >
        {starting ? "STARTING..." : "START RESEARCH \u2192"}
      </button>
    </div>
  );
}
