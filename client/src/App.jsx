import { useReducer, useEffect, useState } from "react";
import { useWebSocket } from "./hooks/useWebSocket";
import { useExperiments } from "./hooks/useExperiments";
import SetupWizard from "./components/SetupWizard";
import Dashboard from "./components/Dashboard";
import History from "./components/History";
import Results from "./components/Results";

const initialState = {
  view: "setup",
  hardware: null,
  session: { active: false, branch: null, startedAt: null, elapsedSeconds: 0 },
  experiments: [],
  agentLog: [],
  currentExperiment: null,
  bestResult: null,
  wsConnected: false,
  error: null,
};

function reducer(state, action) {
  switch (action.type) {
    case "SET_HARDWARE":
      return { ...state, hardware: action.payload };
    case "SET_VIEW":
      return { ...state, view: action.payload };
    case "SET_ERROR":
      return { ...state, error: action.payload };
    case "CLEAR_ERROR":
      return { ...state, error: null };
    case "WS_CONNECTED":
      return { ...state, wsConnected: true };
    case "WS_DISCONNECTED":
      return { ...state, wsConnected: false };

    case "SESSION_STARTED":
      return {
        ...state,
        view: "dashboard",
        session: { active: true, branch: action.payload.branch, startedAt: Date.now(), elapsedSeconds: 0 },
        error: null,
      };
    case "SESSION_STOPPED":
      return { ...state, session: { ...state.session, active: false } };

    case "HEARTBEAT":
      return {
        ...state,
        session: { ...state.session, elapsedSeconds: action.payload.elapsed_seconds },
      };

    case "AGENT_LOG":
      return {
        ...state,
        agentLog: [...state.agentLog.slice(-500), { ...action.payload, timestamp: Date.now() }],
      };

    case "EXPERIMENT_DONE": {
      // Dedup: skip if we already have this experiment (by description + val_bpb)
      const isDupe = state.experiments.some(
        (e) => e.val_bpb === action.payload.val_bpb && e.description === action.payload.description
      );
      if (isDupe) return state;

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
        currentExperiment: {
          n: state.experiments.length + 1,
          hypothesis: action.payload.hypothesis || "",
          ticks: [],
        },
      };

    case "TRAINING_TICK": {
      // If no current experiment, auto-create one
      const current = state.currentExperiment || {
        n: state.experiments.length + 1,
        hypothesis: "",
        ticks: [],
      };
      return {
        ...state,
        currentExperiment: {
          ...current,
          ticks: [
            ...current.ticks.slice(-100),
            { step: action.payload.step, loss: action.payload.loss, elapsed: action.payload.elapsed_seconds },
          ],
        },
      };
    }

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

function formatElapsed(seconds) {
  if (!seconds) return "0:00";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) return `${h}:${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export default function App() {
  const [state, dispatch] = useReducer(reducer, initialState);
  const handleMessage = useExperiments(dispatch);

  const wsUrl = `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/ws`;
  useWebSocket(wsUrl, handleMessage, {
    onOpen: () => dispatch({ type: "WS_CONNECTED" }),
    onClose: () => dispatch({ type: "WS_DISCONNECTED" }),
  });

  useEffect(() => {
    // Load hardware
    fetch("/api/hardware")
      .then((r) => r.json())
      .then((hw) => dispatch({ type: "SET_HARDWARE", payload: hw }))
      .catch(() => dispatch({ type: "SET_ERROR", payload: "Failed to connect to server" }));

    // Load existing experiments (survives server restart)
    fetch("/api/experiments")
      .then((r) => r.json())
      .then((exps) => {
        if (exps?.length > 0) {
          exps.forEach((exp) => dispatch({ type: "EXPERIMENT_DONE", payload: exp }));
          dispatch({ type: "SET_VIEW", payload: "dashboard" });
        }
      })
      .catch(() => {});

    // Check if session is active
    fetch("/api/session/status")
      .then((r) => r.json())
      .then((status) => {
        if (status.active) {
          dispatch({ type: "SESSION_STARTED", payload: { branch: status.branch } });
        }
      })
      .catch(() => {});
  }, []);

  const startSession = async (config) => {
    try {
      const res = await fetch("/api/session/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      return res.json();
    } catch (err) {
      dispatch({ type: "SET_ERROR", payload: err.message });
      throw err;
    }
  };

  const stopSession = () => fetch("/api/session/stop", { method: "POST" });

  return (
    <div className="min-h-screen bg-surface noise-bg">
      {/* Error Toast */}
      {state.error && (
        <div className="fixed top-4 right-4 z-[100] bg-danger-dim border border-danger/30 rounded-lg px-4 py-3 flex items-center gap-3 max-w-sm animate-in">
          <span className="text-danger text-sm">{state.error}</span>
          <button onClick={() => dispatch({ type: "CLEAR_ERROR" })} className="text-danger/60 hover:text-danger text-xs">
            {"\u2715"}
          </button>
        </div>
      )}

      <header className="border-b border-border-dim px-6 py-3 flex items-center justify-between sticky top-0 z-50 bg-surface/90 backdrop-blur-sm">
        <div className="flex items-center gap-3">
          {/* Connection indicator */}
          <div className={`w-2 h-2 rounded-full ${state.wsConnected ? "bg-accent pulse-dot" : "bg-danger"}`} />
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
              className={`px-3 py-1.5 rounded text-[11px] font-mono tracking-widest transition-all relative ${
                state.view === key
                  ? "bg-surface-overlay text-accent border border-border"
                  : "text-text-muted hover:text-text-secondary border border-transparent"
              }`}
            >
              {label}
              {/* Live dot on Dashboard tab when session active */}
              {key === "dashboard" && state.session.active && state.view !== "dashboard" && (
                <span className="absolute -top-0.5 -right-0.5 w-1.5 h-1.5 rounded-full bg-accent pulse-dot" />
              )}
            </button>
          ))}
        </nav>

        <div className="flex items-center gap-4">
          {!state.wsConnected && (
            <span className="font-mono text-[10px] text-danger flex items-center gap-1.5 px-2 py-1 rounded bg-danger-dim border border-danger/20">
              DISCONNECTED
            </span>
          )}
          {state.session.active && (
            <span className="font-mono text-[10px] text-accent flex items-center gap-1.5 px-2 py-1 rounded bg-accent-dim border border-accent/20">
              <span className="w-1.5 h-1.5 rounded-full bg-accent pulse-dot" />
              LIVE {formatElapsed(state.session.elapsedSeconds)}
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
            formatElapsed={formatElapsed}
          />
        )}
        {state.view === "history" && <History experiments={state.experiments} />}
        {state.view === "results" && <Results experiments={state.experiments} />}
      </main>
    </div>
  );
}
