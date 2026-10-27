import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, waitFor, act } from "@testing-library/react";
import { useAgentAttention } from "../../web/src/lib/useAgentAttention";
import { clearToasts, getToasts, setTabAttention } from "../../web/src/lib/attentionEffects";
import { FakeEventSource, mockApi, stubEventSource } from "./helpers";

const agent = (id: string, state: string) => ({ id, name: id, role: "worker", state, pendingApprovals: 0 });
const baseRoutes = (agents: unknown[] = [], chiefState = "idle", approvals: unknown[] = []) => ({
  "GET /api/agents": { body: { agents, max: 3 } },
  "GET /api/chief": { body: { state: chiefState, model: "opus", turns: 0, costUsd: null, sessionId: null, error: null, pendingApprovals: 0, lastSeq: 0 } },
  "GET /api/approvals": { body: { approvals } },
});
const emit = (id: string, state: string, error?: string) =>
  act(() => FakeEventSource.emit({ type: "agent-state", id, name: id, state, ...(error ? { error } : {}) }));

beforeEach(() => stubEventSource());
afterEach(() => {
  vi.unstubAllGlobals();
  clearToasts();
  setTabAttention(0, null); // the module remembers the base title between tests
});

describe("useAgentAttention", () => {
  it("is hidden at first, then turns yellow, then green when an agent finishes with the shelf closed", async () => {
    mockApi(baseRoutes());
    const { result } = renderHook(() => useAgentAttention(false));
    await waitFor(() => expect(FakeEventSource.live).toHaveLength(1));
    expect(result.current.color).toBeNull();
    await emit("a", "running");
    expect(result.current).toMatchObject({ color: "yellow", count: 1 });
    await emit("a", "done");
    expect(result.current).toMatchObject({ color: "green", count: 1 });
    expect(getToasts().map((t) => t.text)).toEqual(["a finished"]);
  });

  it("turns red on failure and clears red/green when the shelf opens", async () => {
    mockApi(baseRoutes());
    const { result, rerender } = renderHook(({ open }) => useAgentAttention(open), { initialProps: { open: false } });
    await waitFor(() => expect(FakeEventSource.live).toHaveLength(1));
    await emit("a", "failed", "boom");
    expect(result.current).toMatchObject({ color: "red", count: 1 });
    rerender({ open: true });
    await waitFor(() => expect(result.current.color).toBeNull());
  });

  it("does not mark unseen or toast when the shelf is open", async () => {
    mockApi(baseRoutes());
    const { result } = renderHook(() => useAgentAttention(true));
    await waitFor(() => expect(FakeEventSource.live).toHaveLength(1));
    await emit("a", "failed");
    expect(result.current.color).toBeNull();
    expect(getToasts()).toEqual([]);
  });

  it("counts pending approvals as red", async () => {
    mockApi(baseRoutes([], "idle", [{ id: "p1" }, { id: "p2" }]));
    const { result } = renderHook(() => useAgentAttention(false));
    await waitFor(() => expect(result.current).toMatchObject({ color: "red", count: 2 }));
  });

  it("seeds running agents and a running chief from the REST lists", async () => {
    mockApi(baseRoutes([agent("a", "running"), agent("b", "done")], "running"));
    const { result } = renderHook(() => useAgentAttention(false));
    await waitFor(() => expect(result.current).toMatchObject({ color: "yellow", count: 2 }));
  });

  it("a seed arriving after a newer event does not resurrect running", async () => {
    mockApi(baseRoutes([agent("a", "running")]));
    const { result } = renderHook(() => useAgentAttention(false));
    await waitFor(() => expect(result.current.count).toBe(1));
    await emit("a", "done");
    expect(result.current).toMatchObject({ color: "green", count: 1 });
  });

  it("updates the tab title with unseen results", async () => {
    document.title = "Hit_finder";
    mockApi(baseRoutes());
    renderHook(() => useAgentAttention(false));
    await waitFor(() => expect(FakeEventSource.live).toHaveLength(1));
    await emit("a", "done");
    await waitFor(() => expect(document.title).toBe("(1) Hit_finder"));
  });
});
