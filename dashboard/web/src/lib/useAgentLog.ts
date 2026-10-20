import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, apiGet } from "./api";
import { subscribe } from "./events";
import type { AgentEvent, AgentEventsResponse } from "../../../shared/types";

const PAGE = 200;
/** The browser keeps this many of the newest entries; the full log stays on disk. */
const MAX_KEPT = 2000;

export interface AgentLog {
  events: AgentEvent[];
  error: string | null;
  loaded: boolean;
}

/**
 * Follows one agent's journal by cursor: asks only for entries after the newest one it holds, and
 * asks again whenever the server announces that agents changed. A reload therefore recovers
 * everything from the server; nothing is buffered here that the server does not have.
 */
export function useAgentLog(id: string, enabled: boolean): AgentLog {
  return useJournal(`/api/agents/${encodeURIComponent(id)}/events`, "agents", enabled);
}

/** The same cursor-following read for any journal endpoint, re-reading when `resource` is announced. */
export function useJournal(basePath: string, resource: string, enabled: boolean): AgentLog {
  const [state, setState] = useState<AgentLog>({ events: [], error: null, loaded: false });
  const cursor = useRef(0);
  const busy = useRef(false);
  const again = useRef(false);
  const live = useRef(true);

  const fetchMore = useCallback(async () => {
    if (busy.current) {
      again.current = true;
      return;
    }
    busy.current = true;
    try {
      for (;;) {
        again.current = false;
        const res = await apiGet<AgentEventsResponse>(`${basePath}?after=${cursor.current}&limit=${PAGE}`);
        if (!live.current) return;
        if (res.lastSeq < cursor.current) {
          // The server's log was replaced (a new conversation): forget ours and read the new one from the top.
          cursor.current = 0;
          setState({ events: [], error: null, loaded: false });
          continue;
        }
        if (res.events.length > 0) {
          cursor.current = res.events[res.events.length - 1].seq;
          setState((s) => ({ events: [...s.events, ...res.events].slice(-MAX_KEPT), error: null, loaded: true }));
        } else {
          setState((s) => (s.loaded && !s.error ? s : { ...s, error: null, loaded: true }));
        }
        if (res.events.length < PAGE && !again.current) return;
      }
    } catch (err) {
      if (live.current) setState((s) => ({ ...s, error: err instanceof ApiError ? err.message : "Could not load the log.", loaded: true }));
    } finally {
      busy.current = false;
    }
  }, [basePath]);

  useEffect(() => {
    live.current = true;
    cursor.current = 0;
    setState({ events: [], error: null, loaded: false });
    if (!enabled) return;
    void fetchMore();
    const off = subscribe({ resource, reload: () => void fetchMore() });
    return () => {
      live.current = false;
      off();
    };
  }, [basePath, resource, enabled, fetchMore]);

  return state;
}
