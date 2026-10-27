import type { AgentState } from "../../../shared/types";

export type BubbleColor = "red" | "yellow" | "green";
export interface Bubble {
  color: BubbleColor | null;
  count: number;
  /** The Agents button's accessible name; always starts with "Agents". */
  label: string;
}

export interface AttentionState {
  /** Agents currently running (not waiting for approval). */
  running: ReadonlySet<string>;
  /** Unseen: failed since the shelf was last opened. */
  failed: ReadonlySet<string>;
  /** Unseen: finished a turn or the whole task since the shelf was last opened. */
  finished: ReadonlySet<string>;
  /** Ids an event has been seen for; a seed from the REST list must not override these. */
  fromEvent: ReadonlySet<string>;
}

export type AttentionAction =
  | { kind: "seed"; running: string[] }
  | { kind: "event"; id: string; state: AgentState; open: boolean }
  | { kind: "open" };

export const EMPTY_ATTENTION: AttentionState = { running: new Set(), failed: new Set(), finished: new Set(), fromEvent: new Set() };

const RUNNING_STATES: AgentState[] = ["starting", "running"];
const FINISHED_STATES: AgentState[] = ["done", "idle"];

const plus = (s: ReadonlySet<string>, id: string): ReadonlySet<string> => new Set([...s, id]);
const minus = (s: ReadonlySet<string>, id: string): ReadonlySet<string> => new Set([...s].filter((x) => x !== id));

/** True when a transition should interrupt a user whose agent shelf is closed. */
export function isUnseenTransition(state: AgentState, open: boolean): boolean {
  return !open && (state === "failed" || FINISHED_STATES.includes(state));
}

export function reduceAttention(s: AttentionState, a: AttentionAction): AttentionState {
  if (a.kind === "open") return { ...s, failed: new Set(), finished: new Set() };
  if (a.kind === "seed") {
    const kept = [...s.running].filter((id) => s.fromEvent.has(id));
    const seeded = a.running.filter((id) => !s.fromEvent.has(id));
    return { ...s, running: new Set([...kept, ...seeded]) };
  }
  const fromEvent = plus(s.fromEvent, a.id);
  if (RUNNING_STATES.includes(a.state)) {
    // A new run starts afresh: drop the old run's unseen marks.
    return { ...s, fromEvent, running: plus(s.running, a.id), failed: minus(s.failed, a.id), finished: minus(s.finished, a.id) };
  }
  const running = minus(s.running, a.id);
  if (a.state === "failed") return { ...s, fromEvent, running, failed: a.open ? s.failed : plus(s.failed, a.id) };
  if (FINISHED_STATES.includes(a.state)) return { ...s, fromEvent, running, finished: a.open ? s.finished : plus(s.finished, a.id) };
  return { ...s, fromEvent, running }; // waiting-approval (counted via approvals) and stopped (the user's own act)
}

/** Unseen results: what the tab title counts. */
export const unseenCount = (s: AttentionState): number => s.failed.size + s.finished.size;

export function deriveBubble(s: AttentionState, approvals: number, open: boolean): Bubble {
  const failed = s.failed.size;
  const running = s.running.size;
  const finished = s.finished.size;
  const parts = [
    failed > 0 ? `${failed} failed` : "",
    approvals > 0 ? `${approvals} waiting for approval` : "",
    running > 0 && !open ? `${running} running` : "",
    finished > 0 ? `${finished} finished` : "",
  ].filter(Boolean);
  const label = parts.length > 0 ? `Agents, ${parts.join(", ")}` : "Agents";
  if (failed + approvals > 0) return { color: "red", count: failed + approvals, label };
  if (!open && running > 0) return { color: "yellow", count: running, label };
  if (finished > 0) return { color: "green", count: finished, label };
  return { color: null, count: 0, label };
}
