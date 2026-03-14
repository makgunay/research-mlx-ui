import { useRef, useEffect } from "react";

export function useWebSocket(url, onMessage) {
  const wsRef = useRef(null);
  const reconnectTimer = useRef(null);
  const onMessageRef = useRef(onMessage);
  onMessageRef.current = onMessage;

  useEffect(() => {
    function connect() {
      wsRef.current = new WebSocket(url);

      wsRef.current.onmessage = (event) => {
        try {
          onMessageRef.current(JSON.parse(event.data));
        } catch (e) {
          console.error("WS parse error", e);
        }
      };

      wsRef.current.onclose = () => {
        reconnectTimer.current = setTimeout(connect, 2000);
      };

      wsRef.current.onerror = () => {};
    }

    connect();
    return () => {
      clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [url]);
}
