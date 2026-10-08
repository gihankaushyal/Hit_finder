import type { DashEvent } from "../../../shared/types";

export const EVENTS_PATH = "/api/events";
export const RECONNECT_MIN_MS = 1000;
export const RECONNECT_MAX_MS = 15_000;
const EVENT_SOURCE_CLOSED = 2;

type Resource = Extract<DashEvent, { type: "invalidate" }>["resource"];
export interface Subscriber {
  resource: string;
  /** Called when the resource was invalidated, or when the stream came back after a drop. */
  reload: () => void;
  /** Called with each line of local test output streamed while a run is active. */
  onLine?: (line: string) => void;
}

// One EventSource shared by every hook; opened for the first subscriber, closed after the last.
const subscribers = new Set<Subscriber>();
let source: EventSource | null = null;
let retryTimer: ReturnType<typeof setTimeout> | null = null;
let retryMs = RECONNECT_MIN_MS;
let dropped = false;

function handleMessage(e: MessageEvent<string> | { data: string }): void {
  let ev: DashEvent;
  try {
    ev = JSON.parse(e.data) as DashEvent;
  } catch {
    return;
  }
  if (ev.type === "tests-line") {
    for (const s of [...subscribers]) s.onLine?.(ev.line);
    return;
  }
  if (ev.type !== "invalidate") return;
  for (const s of [...subscribers]) if (s.resource === (ev.resource as Resource)) s.reload();
}

function connect(): void {
  const es = new EventSource(EVENTS_PATH);
  source = es;
  es.onopen = () => {
    retryMs = RECONNECT_MIN_MS;
    if (dropped) {
      dropped = false;
      for (const s of [...subscribers]) s.reload(); // events may have been missed while offline
    }
  };
  es.onmessage = handleMessage;
  es.onerror = () => {
    dropped = true;
    if (es.readyState !== EVENT_SOURCE_CLOSED) return; // the browser is retrying by itself
    es.close();
    if (source === es) source = null;
    if (subscribers.size === 0 || retryTimer) return;
    retryTimer = setTimeout(() => {
      retryTimer = null;
      if (subscribers.size > 0 && !source) connect();
    }, retryMs);
    retryMs = Math.min(retryMs * 2, RECONNECT_MAX_MS);
  };
}

function shutdown(): void {
  if (retryTimer) clearTimeout(retryTimer);
  retryTimer = null;
  source?.close();
  source = null;
  dropped = false;
  retryMs = RECONNECT_MIN_MS;
}

export function subscribe(s: Subscriber): () => void {
  subscribers.add(s);
  if (!source && !retryTimer) connect();
  return () => {
    subscribers.delete(s);
    if (subscribers.size === 0) shutdown();
  };
}
