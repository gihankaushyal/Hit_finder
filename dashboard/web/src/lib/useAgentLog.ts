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
  const [held, setHeld] = useState<{ path: string; log: AgentLog }>({ path: basePath, log: { events: [], error: null, loaded: false } });
  // What was read from another path is never shown for this one, not even for the render before the reset lands.
  const state: AgentLog = held.path === basePath ? held.log : { events: [], error: null, loaded: false };
  const setState = useCallback((next: AgentLog | ((s: AgentLog) => AgentLog)) => {
    setHeld((h) => (h.path === basePath ? { path: basePath, log: typeof next === "function" ? next(h.log) : next } : h));
  }, [basePath]);
  const generation = useRef(0);
  const cursor = useRef(0);
  const conv = useRef<string | null>(null);
  const busy = useRef(false);
  const again = useRef(false);
  const live = useRef(true);

  const fetchMore = useCallback(async () => {
    if (busy.current) {
      again.current = true;
      return;
    }
    busy.current = true;
    const mine = generation.current;
    const stale = () => !live.current || generation.current !== mine;
    try {
      for (;;) {
        again.current = false;
        const res = await apiGet<AgentEventsResponse>(`${basePath}?after=${cursor.current}&limit=${PAGE}`);
        if (stale()) return; // the path changed while this was in flight: its answer belongs to another log
        const switched = res.conversationId !== undefined && conv.current !== null && res.conversationId !== conv.current;
        if (res.conversationId !== undefined) conv.current = res.conversationId;
        if (switched || res.lastSeq < cursor.current) {
          // The server's log was replaced (a new or resumed conversation): forget ours and read the new one from the top.
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
      if (!stale()) setState((s) => ({ ...s, error: err instanceof ApiError ? err.message : "Could not load the log.", loaded: true }));
    } finally {
      if (generation.current === mine) busy.current = false;
    }
  }, [basePath, setState]);

  useEffect(() => {
    live.current = true;
    generation.current++; // abandons any fetch still running for the previous path
    busy.current = false;
    again.current = false;
    cursor.current = 0;
    conv.current = null;
    setHeld({ path: basePath, log: { events: [], error: null, loaded: false } });
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
