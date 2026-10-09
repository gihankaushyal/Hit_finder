import { useCallback, useRef, useSyncExternalStore } from "react";
import { ApiError, apiGet } from "./api";
import { subscribe as subscribeEvents } from "./events";

export interface ResourceState<T> {
  data: T | null;
  error: string | null;
  /** True only until the first answer; a reload keeps showing the previous data. */
  loading: boolean;
  /** Loaded fine, and there is nothing to show. */
  empty: boolean;
  /** The server rejected the cookie: the user must open the tokenised URL again. */
  unauthorized: boolean;
  reload: () => void;
}

/** The server never announces CI, PR or branch changes, so these resources are polled. */
export const CI_REFRESH_MS = 60_000;
export const PRS_REFRESH_MS = 60_000;
export const REPO_REFRESH_MS = 30_000;

const defaultIsEmpty = (d: unknown): boolean => Array.isArray(d) && d.length === 0;

interface Snapshot<T> {
  data: T | null;
  error: string | null;
  unauthorized: boolean;
  loading: boolean;
}
const INITIAL: Snapshot<never> = { data: null, error: null, unauthorized: false, loading: true };

interface Member {
  notify: () => void;
  refreshMs: number | undefined;
}
/** One request, one cached value and one timer per path, however many components read it. */
interface Entry {
  path: string;
  resource: string;
  snap: Snapshot<unknown>;
  members: Set<Member>;
  latest: number;
  dead: boolean;
  timer: ReturnType<typeof setInterval> | null;
  timerMs: number | undefined;
  unsubscribeEvents: () => void;
  load: () => void;
  retime: () => void;
  onVisibility: () => void;
}

const entries = new Map<string, Entry>();

const isHidden = (): boolean => typeof document !== "undefined" && document.visibilityState === "hidden";

function createEntry(path: string, resource: string): Entry {
  const entry: Entry = {
    path, resource, snap: INITIAL, members: new Set(), latest: 0, dead: false, timer: null, timerMs: undefined,
    unsubscribeEvents: () => {},
    load: () => {},
    retime: () => {},
    onVisibility: () => {},
  };
  const publish = (snap: Snapshot<unknown>) => {
    entry.snap = snap;
    for (const m of [...entry.members]) m.notify();
  };
  entry.load = () => {
    const id = ++entry.latest;
    apiGet<unknown>(path).then(
      (d) => {
        if (entry.dead || id !== entry.latest) return;
        publish({ data: d, error: null, unauthorized: false, loading: false });
      },
      (err: unknown) => {
        if (entry.dead || id !== entry.latest) return;
        publish({
          ...entry.snap,
          error: err instanceof ApiError ? err.message : "Request failed",
          unauthorized: err instanceof ApiError && err.status === 401,
          loading: false,
        });
      },
    );
  };
  entry.retime = () => {
    const wanted = [...entry.members].map((m) => m.refreshMs).filter((ms): ms is number => ms !== undefined);
    const ms = wanted.length > 0 ? Math.min(...wanted) : undefined;
    if (entry.timer && ms === entry.timerMs) return;
    if (entry.timer) clearInterval(entry.timer);
    entry.timer = null;
    entry.timerMs = ms;
    if (ms !== undefined && !isHidden()) entry.timer = setInterval(entry.load, ms);
  };
  entry.onVisibility = () => {
    if (isHidden()) {
      if (entry.timer) clearInterval(entry.timer);
      entry.timer = null;
      return;
    }
    if (entry.timerMs === undefined) return;
    entry.load(); // the tab was away: catch up now, then resume the cadence
    entry.timer = entry.timer ?? setInterval(entry.load, entry.timerMs);
  };
  return entry;
}

function acquire(path: string, resource: string, member: Member): () => void {
  let entry = entries.get(path);
  const fresh = !entry;
  if (!entry) {
    entry = createEntry(path, resource);
    entries.set(path, entry);
  }
  entry.members.add(member);
  if (fresh) {
    entry.unsubscribeEvents = subscribeEvents({ resource, reload: entry.load });
    document.addEventListener("visibilitychange", entry.onVisibility);
    entry.load();
  }
  entry.retime();
  const e = entry;
  return () => {
    e.members.delete(member);
    if (e.members.size > 0) {
      e.retime();
      return;
    }
    e.dead = true;
    if (e.timer) clearInterval(e.timer);
    e.timer = null;
    e.unsubscribeEvents();
    document.removeEventListener("visibilitychange", e.onVisibility);
    if (entries.get(path) === e) entries.delete(path);
  };
}

/**
 * Reads `path` through a store shared by every component that asks for it, and fetches again whenever the
 * server announces that `resource` changed (or the event stream came back after a drop). With `refreshMs`
 * it also polls, but only while the tab is visible.
 */
export function useResource<T>(
  resource: string,
  path: string,
  isEmpty: (d: T) => boolean = defaultIsEmpty,
  refreshMs?: number,
): ResourceState<T> {
  const isEmptyRef = useRef(isEmpty);
  isEmptyRef.current = isEmpty;
  const subscribe = useCallback(
    (notify: () => void) => acquire(path, resource, { notify, refreshMs }),
    [path, resource, refreshMs],
  );
  const getSnapshot = useCallback(() => (entries.get(path)?.snap ?? INITIAL) as Snapshot<T>, [path]);
  const snap = useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
  const reload = useCallback(() => entries.get(path)?.load(), [path]);
  return {
    data: snap.data,
    error: snap.error,
    loading: snap.loading,
    unauthorized: snap.unauthorized,
    empty: snap.data !== null && snap.error === null && isEmptyRef.current(snap.data),
    reload,
  };
}
