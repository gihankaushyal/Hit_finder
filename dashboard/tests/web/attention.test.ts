import { describe, it, expect } from "vitest";
import { EMPTY_ATTENTION, deriveBubble, isUnseenTransition, reduceAttention, unseenCount, type AttentionState } from "../../web/src/lib/attention";

const run = (s: AttentionState, id: string, state: string, open = false) =>
  reduceAttention(s, { kind: "event", id, state: state as never, open });

describe("reduceAttention", () => {
  it("tracks running agents and drops them when they finish", () => {
    let s = run(EMPTY_ATTENTION, "a", "running");
    expect([...s.running]).toEqual(["a"]);
    s = run(s, "a", "done");
    expect(s.running.size).toBe(0);
    expect([...s.finished]).toEqual(["a"]);
  });

  it("marks failures and finishes unseen only while the shelf is closed", () => {
    expect(run(EMPTY_ATTENTION, "a", "failed", true).failed.size).toBe(0);
    expect(run(EMPTY_ATTENTION, "a", "idle", true).finished.size).toBe(0);
    expect(run(EMPTY_ATTENTION, "a", "failed", false).failed.has("a")).toBe(true);
    expect(run(EMPTY_ATTENTION, "a", "idle", false).finished.has("a")).toBe(true);
  });

  it("ignores stopped and keeps waiting-approval out of running", () => {
    const s = run(run(EMPTY_ATTENTION, "a", "running"), "a", "waiting-approval");
    expect(s.running.size).toBe(0);
    const t = run(run(EMPTY_ATTENTION, "b", "running"), "b", "stopped");
    expect(t.running.size).toBe(0);
    expect(t.finished.size + t.failed.size).toBe(0);
  });

  it("a new run of an agent clears its old unseen marks", () => {
    let s = run(EMPTY_ATTENTION, "a", "failed");
    s = run(s, "a", "running");
    expect(s.failed.size).toBe(0);
    expect(s.running.has("a")).toBe(true);
  });

  it("clears unseen marks when the shelf opens but keeps running", () => {
    let s = run(run(EMPTY_ATTENTION, "a", "failed"), "b", "running");
    s = run(s, "c", "done");
    s = reduceAttention(s, { kind: "open" });
    expect(s.failed.size + s.finished.size).toBe(0);
    expect(s.running.has("b")).toBe(true);
  });

  it("a seed adds running ids but cannot resurrect an agent that already reported", () => {
    let s = run(EMPTY_ATTENTION, "a", "done");
    s = reduceAttention(s, { kind: "seed", running: ["a", "b"], settled: [] });
    expect([...s.running]).toEqual(["b"]);
  });

  it("a later seed replaces earlier seeded ids but keeps event-driven ones", () => {
    let s = reduceAttention(EMPTY_ATTENTION, { kind: "seed", running: ["x"], settled: [] });
    s = run(s, "y", "running");
    s = reduceAttention(s, { kind: "seed", running: [], settled: [] });
    expect([...s.running]).toEqual(["y"]);
  });

  it("a seed that lists an event-driven id as settled clears it (a missed done must not stick the bubble yellow)", () => {
    let s = run(EMPTY_ATTENTION, "y", "running");
    s = reduceAttention(s, { kind: "seed", running: [], settled: ["y"] });
    expect(s.running.size).toBe(0);
  });
});

describe("deriveBubble", () => {
  const withSets = (p: Partial<AttentionState>): AttentionState => ({ ...EMPTY_ATTENTION, ...p });
  it("is hidden when nothing needs attention", () => {
    expect(deriveBubble(EMPTY_ATTENTION, 0, false)).toEqual({ color: null, count: 0, label: "Agents" });
  });
  it("red wins: failed agents plus approvals", () => {
    const b = deriveBubble(withSets({ failed: new Set(["a", "b"]), running: new Set(["c"]), finished: new Set(["d"]) }), 1, false);
    expect(b).toEqual({ color: "red", count: 3, label: "Agents, 2 failed, 1 waiting for approval, 1 running, 1 finished" });
  });
  it("yellow shows running only while the shelf is closed", () => {
    const s = withSets({ running: new Set(["a", "b"]) });
    expect(deriveBubble(s, 0, false)).toMatchObject({ color: "yellow", count: 2 });
    expect(deriveBubble(s, 0, true).color).toBeNull();
  });
  it("green counts unseen finished agents", () => {
    expect(deriveBubble(withSets({ finished: new Set(["a"]) }), 0, false)).toEqual({ color: "green", count: 1, label: "Agents, 1 finished" });
  });
});

describe("helpers", () => {
  it("unseenCount is failed plus finished", () => {
    expect(unseenCount({ ...EMPTY_ATTENTION, failed: new Set(["a"]), finished: new Set(["b", "c"]) })).toBe(3);
  });
  it("isUnseenTransition is true for failed/done/idle with the shelf closed", () => {
    expect(isUnseenTransition("failed", false)).toBe(true);
    expect(isUnseenTransition("done", false)).toBe(true);
    expect(isUnseenTransition("idle", false)).toBe(true);
    expect(isUnseenTransition("idle", true)).toBe(false);
    expect(isUnseenTransition("running", false)).toBe(false);
    expect(isUnseenTransition("stopped", false)).toBe(false);
  });
});
