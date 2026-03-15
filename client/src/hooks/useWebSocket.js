import { useRef, useEffect } from "react";

export function useWebSocket(url, onMessage, { onOpen, onClose } = {}) {
  const wsRef = useRef(null);
  const reconnectTimer = useRef(null);
  const onMessageRef = useRef(onMessage);
  const onOpenRef = useRef(onOpen);
  const onCloseRef = useRef(onClose);
  onMessageRef.current = onMessage;
  onOpenRef.current = onOpen;
  onCloseRef.current = onClose;

  useEffect(() => {
    let disposed = false;

    function connect() {
      if (disposed) return;
      wsRef.current = new WebSocket(url);

      wsRef.current.onopen = () => {
        onOpenRef.current?.();
      };

      wsRef.current.onmessage = (event) => {
        try {
          onMessageRef.current(JSON.parse(event.data));
        } catch (e) {
          console.error("WS parse error", e);
        }
      };

      wsRef.current.onclose = () => {
        if (disposed) return;
        onCloseRef.current?.();
        reconnectTimer.current = setTimeout(connect, 2000);
      };

      wsRef.current.onerror = () => {};
    }

    connect();
    return () => {
      disposed = true;
      clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [url]);
}
