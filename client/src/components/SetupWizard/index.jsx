import { useState, useEffect } from "react";

const FOCUS_OPTIONS = [
  { key: "speed", label: "Speed", icon: "\u26A1", desc: "Maximize throughput" },
  { key: "memory", label: "Memory", icon: "\u2B1B", desc: "Fit larger models" },
  { key: "accuracy", label: "Accuracy", icon: "\u25CE", desc: "Push BPB lower" },
  { key: "optimizer", label: "Optimizer", icon: "\u2699\uFE0F", desc: "Tune Muon/AdamW" },
];

export default function SetupWizard({ hardware, onStart, onProjectSwitch }) {
  const [projects, setProjects] = useState([]);
  const [focus, setFocus] = useState([]);
  const [hints, setHints] = useState("");
  const [maxExperiments, setMaxExperiments] = useState(15);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState(null);
  const [showNewProject, setShowNewProject] = useState(false);
  const [newProjectName, setNewProjectName] = useState("");
  const [forkFrom, setForkFrom] = useState("");

  const activeProject = projects.find((p) => p.active);

  useEffect(() => {
    fetch("/api/projects")
      .then((r) => r.json())
      .then(setProjects)
      .catch(() => {});
  }, []);

  const toggleFocus = (key) =>
    setFocus((f) => (f.includes(key) ? f.filter((k) => k !== key) : [...f, key]));

  const handleStart = async () => {
    setStarting(true);
    setError(null);
    try {
      await onStart({
        focusAreas: focus,
        hints,
        maxExperiments,
        branchName: new Date().toISOString().slice(0, 10),
      });
    } catch (err) {
      setError(err.message || "Failed to start session");
      setStarting(false);
    }
  };

  const handleCreateProject = async () => {
    setError(null);
    try {
      const res = await fetch("/api/projects", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: newProjectName, forkFrom: forkFrom || null }),
      });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || "Failed to create project");
      }
      setShowNewProject(false);
      setNewProjectName("");
      setForkFrom("");
      const updated = await fetch("/api/projects").then((r) => r.json());
      setProjects(updated);
      onProjectSwitch?.();
    } catch (err) {
      setError(err.message);
    }
  };

  const handleSwitchProject = async (name) => {
    setError(null);
    try {
      const res = await fetch(`/api/projects/${name}/activate`, { method: "POST" });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || "Failed to switch project");
      }
      const updated = await fetch("/api/projects").then((r) => r.json());
      setProjects(updated);
      onProjectSwitch?.();
    } catch (err) {
      setError(err.message);
    }
  };

  const handleDeleteProject = async (name) => {
    if (!confirm(`Delete project "${name}"? This removes its results and train.py snapshot.`)) return;
    setError(null);
    try {
      const res = await fetch(`/api/projects/${name}`, { method: "DELETE" });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || "Failed to delete project");
      }
      const updated = await fetch("/api/projects").then((r) => r.json());
      setProjects(updated);
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div className="max-w-xl mx-auto space-y-6 pt-8">
      {/* Title */}
      <div className="text-center space-y-2 mb-6">
        <h1 className="text-2xl font-mono tracking-tight text-text-primary">
          Research Session
        </h1>
        <p className="text-sm text-text-muted">
          Configure and launch autonomous ML experiments
        </p>
      </div>

      {/* Projects */}
      <div className="bg-surface-raised border border-border-dim rounded-xl p-5 space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-[10px] font-mono tracking-widest text-text-muted">PROJECTS</span>
          <button
            type="button"
            onClick={() => setShowNewProject(!showNewProject)}
            className="text-[10px] font-mono text-accent hover:text-accent/80 transition-colors"
          >
            {showNewProject ? "CANCEL" : "+ NEW PROJECT"}
          </button>
        </div>

        {/* New Project Form */}
        {showNewProject && (
          <div className="bg-surface-overlay border border-border-dim rounded-lg p-4 space-y-3">
            <input
              value={newProjectName}
              onChange={(e) => setNewProjectName(e.target.value.toLowerCase().replace(/[^a-z0-9-]/g, "").replace(/^-+/, ""))}
              placeholder="project-name"
              className="w-full bg-surface border border-border-dim rounded-lg px-3 py-2 text-sm font-mono text-text-primary placeholder-text-muted focus:outline-none focus:border-accent/50"
            />
            <div className="space-y-1">
              <span className="text-[10px] font-mono text-text-muted">START FROM</span>
              <div className="space-y-1">
                <label className="flex items-center gap-2 text-sm text-text-secondary cursor-pointer">
                  <input
                    type="radio"
                    name="fork"
                    checked={!forkFrom}
                    onChange={() => setForkFrom("")}
                    className="accent-accent"
                  />
                  Baseline (original train.py)
                </label>
                {projects.map((p) => (
                  <label key={p.name} className="flex items-center gap-2 text-sm text-text-secondary cursor-pointer">
                    <input
                      type="radio"
                      name="fork"
                      checked={forkFrom === p.name}
                      onChange={() => setForkFrom(p.name)}
                      className="accent-accent"
                    />
                    Fork: {p.name}
                    {p.best_bpb && (
                      <span className="text-[10px] font-mono text-accent ml-auto">
                        {p.best_bpb.toFixed(4)} BPB
                      </span>
                    )}
                  </label>
                ))}
              </div>
            </div>
            <button
              type="button"
              onClick={handleCreateProject}
              disabled={!newProjectName || newProjectName.length < 2}
              className="w-full py-2 rounded-lg bg-accent text-surface text-sm font-mono font-semibold disabled:opacity-40"
            >
              CREATE PROJECT
            </button>
          </div>
        )}

        {/* Project List */}
        {projects.length === 0 && !showNewProject && (
          <div className="text-sm text-text-muted py-2">
            No projects yet. Create one to get started.
          </div>
        )}
        <div className="space-y-1">
          {projects.map((p) => (
            <div
              key={p.name}
              role={p.active ? undefined : "button"}
              tabIndex={p.active ? undefined : 0}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all ${
                p.active
                  ? "bg-accent-dim border border-accent/20"
                  : "bg-surface-overlay border border-border-dim hover:border-border cursor-pointer"
              }`}
              onClick={() => !p.active && handleSwitchProject(p.name)}
              onKeyDown={(e) => { if (!p.active && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); handleSwitchProject(p.name); } }}
            >
              <span className={`w-2 h-2 rounded-full flex-shrink-0 ${p.active ? "bg-accent" : "bg-discard"}`} />
              <span className={`text-sm font-mono flex-1 ${p.active ? "text-accent" : "text-text-secondary"}`}>
                {p.name}
              </span>
              <span className="text-[11px] font-mono text-text-muted tabular-nums">
                {p.experiments > 0 ? `${p.experiments} exps` : "new"}
              </span>
              {p.best_bpb && (
                <span className="text-[11px] font-mono text-accent tabular-nums">
                  {p.best_bpb.toFixed(4)}
                </span>
              )}
              {!p.active && p.experiments === 0 && (
                <button
                  type="button"
                  onClick={(e) => { e.stopPropagation(); handleDeleteProject(p.name); }}
                  className="text-text-muted hover:text-danger text-xs transition-colors"
                >
                  {"\u2715"}
                </button>
              )}
              {p.active && (
                <span className="text-[10px] font-mono text-accent">ACTIVE</span>
              )}
            </div>
          ))}
        </div>
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
      </div>

      {/* Experiment Limit */}
      <div className="bg-surface-raised border border-border-dim rounded-xl p-5 space-y-3">
        <span className="text-[10px] font-mono tracking-widest text-text-muted">SESSION LENGTH</span>
        <div className="flex items-center gap-4">
          <input
            type="range"
            min={5}
            max={50}
            step={5}
            value={maxExperiments}
            onChange={(e) => setMaxExperiments(Number(e.target.value))}
            className="flex-1 accent-accent"
          />
          <span className="font-mono text-sm text-text-primary w-20 text-right tabular-nums">
            {maxExperiments} exps
          </span>
        </div>
        <p className="text-[11px] text-text-muted">
          ~{maxExperiments * 6} min
          {" \u2014 "}
          {maxExperiments <= 10 ? "Quick sprint" :
           maxExperiments <= 20 ? "Standard session" :
           maxExperiments <= 35 ? "Deep exploration" : "Marathon run"}
        </p>
      </div>

      {/* Hints */}
      <div className="bg-surface-raised border border-border-dim rounded-xl p-5 space-y-3">
        <span className="text-[10px] font-mono tracking-widest text-text-muted">HINTS (OPTIONAL)</span>
        <textarea
          value={hints}
          onChange={(e) => setHints(e.target.value)}
          placeholder="e.g. 'try grouped query attention' or 'explore gradient clipping'"
          className="w-full h-20 bg-surface-overlay border border-border-dim rounded-lg p-3 text-sm text-text-primary placeholder-text-muted resize-none focus:outline-none focus:border-accent/50 transition-colors"
        />
      </div>

      {/* Error */}
      {error && (
        <div className="bg-danger-dim border border-danger/30 rounded-xl px-4 py-3 text-sm text-danger font-mono">
          {error}
        </div>
      )}

      {/* Start Button */}
      <button
        onClick={handleStart}
        disabled={!hardware || !activeProject || starting}
        className="w-full py-3.5 rounded-xl font-mono text-sm tracking-wider bg-accent text-surface font-semibold hover:bg-accent/90 transition-all disabled:opacity-40 disabled:cursor-not-allowed glow-green flex items-center justify-center gap-2"
      >
        {starting && (
          <span className="inline-block w-4 h-4 border-2 border-surface/40 border-t-surface rounded-full animate-spin" />
        )}
        {starting
          ? "STARTING..."
          : activeProject
            ? `START ${activeProject.name.toUpperCase()} (${maxExperiments} exps) \u2192`
            : "SELECT A PROJECT FIRST"}
      </button>
    </div>
  );
}
