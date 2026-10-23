import { useCallback, useSyncExternalStore } from "react";

const CHANGE_EVENT = "dash-querychange";

type UpdateMode = "push" | "replace";

function subscribe(onChange: () => void): () => void {
  window.addEventListener("popstate", onChange);
  window.addEventListener(CHANGE_EVENT, onChange);
  return () => {
    window.removeEventListener("popstate", onChange);
    window.removeEventListener(CHANGE_EVENT, onChange);
  };
}

const getSearch = () => window.location.search;
const getServerSearch = () => "";

/** Writes query params without a router. A null value removes the param. */
export function setQueryParams(updates: Record<string, string | null>, mode: UpdateMode = "replace"): void {
  const url = new URL(window.location.href);
  for (const [key, value] of Object.entries(updates)) {
    if (value === null) url.searchParams.delete(key);
    else url.searchParams.set(key, value);
  }
  const next = `${url.pathname}${url.search}${url.hash}`;
  if (next === `${window.location.pathname}${window.location.search}${window.location.hash}`) return;
  if (mode === "push") window.history.pushState(null, "", next);
  else window.history.replaceState(null, "", next);
  window.dispatchEvent(new Event(CHANGE_EVENT));
}

/** The whole query string, kept current as it changes. */
export function useSearch(): string {
  return useSyncExternalStore(subscribe, getSearch, getServerSearch);
}

/** The address of a tab with every other param (filters, sort) carried over, so a new window opens the same view. */
export function tabHref(search: string, tab: string, defaultTab: string): string {
  const params = new URLSearchParams(search);
  if (tab === defaultTab) params.delete("tab");
  else params.set("tab", tab);
  const qs = params.toString();
  return qs ? `?${qs}` : "?";
}

/** One query param as state. The default value is kept out of the URL so addresses stay short. */
export function useQueryParam(key: string, fallback: string): [string, (value: string, mode?: UpdateMode) => void] {
  const search = useSearch();
  const value = new URLSearchParams(search).get(key) ?? fallback;
  const set = useCallback(
    (next: string, mode: UpdateMode = "replace") => setQueryParams({ [key]: next === fallback ? null : next }, mode),
    [key, fallback],
  );
  return [value, set];
}
