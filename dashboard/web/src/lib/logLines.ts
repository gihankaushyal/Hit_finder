/** A line with a stable id, so a sliding window of output can keep its rows mounted. */
export interface LogLine {
  id: number;
  text: string;
}

/** Where `needle` occurs as a contiguous run in `hay`, or -1. */
function indexOfRun(hay: LogLine[], needle: string[]): number {
  for (let from = 0; from + needle.length <= hay.length; from++) {
    let same = true;
    for (let i = 0; i < needle.length && same; i++) same = hay[from + i].text === needle[i];
    if (same) return from;
  }
  return -1;
}

/**
 * Fold the server's snapshot of the output into what is shown. Rows already on screen keep their
 * ids, so a reload remounts nothing it does not have to:
 * - an empty snapshot clears the output (a new run has started);
 * - a snapshot that is already part of what is shown changes nothing, since it can only lag the stream;
 * - one that continues what is shown keeps the overlap and adds the rest;
 * - anything else replaces it.
 */
export function reconcile(prev: LogLine[], next: string[], newId: () => number): LogLine[] {
  if (next.length === 0) return [];
  if (indexOfRun(prev, next) >= 0) return prev;
  // Find the earliest point in `prev` from which everything that follows matches the start of `next`.
  for (let from = 0; from < prev.length; from++) {
    const overlap = prev.length - from;
    if (overlap > next.length) continue;
    let same = true;
    for (let i = 0; i < overlap && same; i++) same = prev[from + i].text === next[i];
    if (!same) continue;
    return [...prev.slice(from), ...next.slice(overlap).map((text) => ({ id: newId(), text }))];
  }
  return next.map((text) => ({ id: newId(), text }));
}
