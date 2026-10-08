import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, apiGet } from "./api";
import { subscribe } from "./events";

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

const defaultIsEmpty = (d: unknown): boolean => Array.isArray(d) && d.length === 0;

/**
 * Fetches `path`, and fetches again whenever the server announces that `resource` changed
 * (or the event stream came back after a drop).
 */
export function useResource<T>(resource: string, path: string, isEmpty: (d: T) => boolean = defaultIsEmpty): ResourceState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [unauthorized, setUnauthorized] = useState(false);
  const [loading, setLoading] = useState(true);
  const latest = useRef(0);
  const alive = useRef(true);
  const isEmptyRef = useRef(isEmpty);
  isEmptyRef.current = isEmpty;

  const load = useCallback(() => {
    const id = ++latest.current;
    apiGet<T>(path).then(
      (d) => {
        if (!alive.current || id !== latest.current) return;
        setData(d);
        setError(null);
        setUnauthorized(false);
        setLoading(false);
      },
      (err: unknown) => {
        if (!alive.current || id !== latest.current) return;
        setError(err instanceof ApiError ? err.message : "Request failed");
        setUnauthorized(err instanceof ApiError && err.status === 401);
        setLoading(false);
      },
    );
  }, [path]);

  useEffect(() => {
    alive.current = true;
    load();
    const unsubscribe = subscribe({ resource, reload: load });
    return () => {
      alive.current = false;
      latest.current++;
      unsubscribe();
    };
  }, [resource, load]);

  return { data, error, loading, empty: data !== null && error === null && isEmptyRef.current(data), unauthorized, reload: load };
}
