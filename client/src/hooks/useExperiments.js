import { useCallback } from "react";

export function useExperiments(dispatch) {
  const handleMessage = useCallback((msg) => {
    switch (msg.type) {
      case "session_started":
        dispatch({ type: "SESSION_STARTED", payload: msg });
        break;
      case "session_stopped":
        dispatch({ type: "SESSION_STOPPED", payload: msg });
        break;
      case "agent_log":
        dispatch({ type: "AGENT_LOG", payload: msg });
        break;
      case "experiment_done":
        dispatch({ type: "EXPERIMENT_DONE", payload: msg });
        break;
      case "training_tick":
        dispatch({ type: "TRAINING_TICK", payload: msg });
        break;
      case "experiment_started":
        dispatch({ type: "EXPERIMENT_STARTED", payload: msg });
        break;
      case "heartbeat":
        dispatch({ type: "HEARTBEAT", payload: msg });
        break;
      case "error":
        dispatch({ type: "AGENT_LOG", payload: { level: "error", text: msg.message } });
        break;
    }
  }, [dispatch]);

  return handleMessage;
}
