/** A line with a stable id, so a sliding window of output can keep its rows mounted. */
export interface LogLine {
  id: number;
  text: string;
}

/**
 * The server's snapshot of the output, as lines. Rows that are already on screen keep their ids,
 * so a reload that changes nothing (or only slides the window) remounts nothing.
 */
export function reconcile(prev: LogLine[], next: string[], newId: () => number): LogLine[] {
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

/**
 * Streamed lines the snapshot may already contain: when the end of the snapshot equals the start of the
 * buffered lines, those lines are in it. Returns the buffered lines that are not.
 */
export function dropSeen(snapshot: string[], buffered: LogLine[]): LogLine[] {
  const most = Math.min(snapshot.length, buffered.length);
  for (let k = most; k > 0; k--) {
    let same = true;
    for (let i = 0; i < k && same; i++) same = snapshot[snapshot.length - k + i] === buffered[i].text;
    if (same) return buffered.slice(k);
  }
  return buffered;
}
