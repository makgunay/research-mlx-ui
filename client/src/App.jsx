import { useReducer, useEffect } from "react";
import { useWebSocket } from "./hooks/useWebSocket";
import { useExperiments } from "./hooks/useExperiments";
import SetupWizard from "./components/SetupWizard";
import Dashboard from "./components/Dashboard";
import History from "./components/History";
import Results from "./components/Results";

const initialState = {
  view: "setup",
  hardware: null,
  session: { active: false, branch: null, startedAt: null },
  experiments: [],
  agentLog: [],
  currentExperiment: null,
  bestResult: null,
};

function reducer(state, action) {
  switch (action.type) {
    case "SET_HARDWARE":
      return { ...state, hardware: action.payload };
    case "SET_VIEW":
      return { ...state, view: action.payload };
    case "SESSION_STARTED":
      return {
        ...state,
        view: "dashboard",
        session: { active: true, branch: action.payload.branch, startedAt: Date.now() },
      };
    case "SESSION_STOPPED":
      return { ...state, session: { ...state.session, active: false } };
    case "AGENT_LOG":
      return {
        ...state,
        agentLog: [...state.agentLog.slice(-500), { ...action.payload, timestamp: Date.now() }],
      };
    case "EXPERIMENT_DONE": {
      const exp = { ...action.payload, n: state.experiments.length + 1 };
      const experiments = [...state.experiments, exp];
      const kept = experiments.filter((e) => e.status === "keep");
      const bestResult = kept.length
        ? kept.reduce((a, b) => (a.val_bpb < b.val_bpb ? a : b))
        : state.bestResult;
      return { ...state, experiments, bestResult, currentExperiment: null };
    }
    case "EXPERIMENT_STARTED":
      return {
        ...state,
        currentExperiment: { n: state.experiments.length + 1, hypothesis: action.payload.hypothesis || "", ticks: [] },
      };
    case "TRAINING_TICK":
      if (!state.currentExperiment) return state;
      return {
        ...state,
        currentExperiment: {
          ...state.currentExperiment,
          ticks: [...state.currentExperiment.ticks, { step: action.payload.step, loss: action.payload.loss, elapsed: action.payload.elapsed_seconds }],
        },
      };
    default:
      return state;
  }
}

const NAV = [
  { key: "setup", label: "SETUP" },
  { key: "dashboard", label: "DASHBOARD" },
  { key: "history", label: "HISTORY" },
  { key: "results", label: "RESULTS" },
];

export default function App() {
  const [state, dispatch] = useReducer(reducer, initialState);
  const handleMessage = useExperiments(dispatch);

  const wsUrl = `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/ws`;
  useWebSocket(wsUrl, handleMessage);

  useEffect(() => {
    fetch("/api/hardware")
      .then((r) => r.json())
      .then((hw) => dispatch({ type: "SET_HARDWARE", payload: hw }))
      .catch(() => {});
  }, []);

  const startSession = async (config) => {
    const res = await fetch("/api/session/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    });
    return res.json();
  };

  const stopSession = () => fetch("/api/session/stop", { method: "POST" });

  return (
    <div className="min-h-screen bg-surface noise-bg">
      <header className="border-b border-border-dim px-6 py-3 flex items-center justify-between sticky top-0 z-50 bg-surface/90 backdrop-blur-sm">
        <div className="flex items-center gap-3">
          <div className="w-2 h-2 rounded-full bg-accent pulse-dot" />
          <span className="font-mono text-sm tracking-[0.2em] text-text-secondary font-medium">
            AUTORESEARCH
          </span>
          <span className="font-mono text-[10px] px-1.5 py-0.5 rounded bg-accent-dim text-accent border border-accent/20">
            MLX
          </span>
        </div>
        <nav className="flex gap-0.5">
          {NAV.map(({ key, label }) => (
            <button
              key={key}
              onClick={() => dispatch({ type: "SET_VIEW", payload: key })}
              className={`px-3 py-1.5 rounded text-[11px] font-mono tracking-widest transition-all ${
                state.view === key
                  ? "bg-surface-overlay text-accent border border-border"
                  : "text-text-muted hover:text-text-secondary border border-transparent"
              }`}
            >
              {label}
            </button>
          ))}
        </nav>
        <div className="flex items-center gap-4">
          {state.session.active && (
            <span className="font-mono text-[10px] text-accent flex items-center gap-1.5 px-2 py-1 rounded bg-accent-dim border border-accent/20">
              <span className="w-1.5 h-1.5 rounded-full bg-accent pulse-dot" />
              LIVE
            </span>
          )}
          {state.hardware && (
            <span className="font-mono text-[10px] text-text-muted tracking-wide">
              {state.hardware.chip} / {state.hardware.memory}
            </span>
          )}
        </div>
      </header>

      <main className="p-6 max-w-[1600px] mx-auto">
        {state.view === "setup" && <SetupWizard hardware={state.hardware} onStart={startSession} />}
        {state.view === "dashboard" && (
          <Dashboard
            session={state.session}
            experiments={state.experiments}
            agentLog={state.agentLog}
            currentExperiment={state.currentExperiment}
            bestResult={state.bestResult}
            onStop={stopSession}
          />
        )}
        {state.view === "history" && <History experiments={state.experiments} />}
        {state.view === "results" && <Results experiments={state.experiments} />}
      </main>
    </div>
  );
}
