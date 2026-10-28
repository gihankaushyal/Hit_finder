import { describe, it, expect, afterEach, beforeEach, vi } from "vitest";
import { act, renderHook, waitFor } from "@testing-library/react";
import { useJournal } from "../../web/src/lib/useAgentLog";
import { FakeEventSource, mockApi, stubEventSource } from "./helpers";
import type { AgentEvent, AgentEventsResponse } from "../../shared/types";

beforeEach(() => stubEventSource());
afterEach(() => vi.unstubAllGlobals());

const ev = (seq: number, text: string): AgentEvent => ({ kind: "user", text, seq, at: "2026-10-20T08:01:00.000Z" });
const page = (events: AgentEvent[], conversationId: string): { body: AgentEventsResponse } => ({
  body: { events, lastSeq: events.length > 0 ? events[events.length - 1].seq : 0, conversationId },
});

describe("useJournal", () => {
  it("ignores a slow answer for the path it has left", async () => {
    let release: () => void = () => undefined;
    const gate = new Promise<void>((r) => (release = r));
    mockApi({
      "GET /api/chief/events?after=0&limit=200": async () => (await gate, page([ev(1, "live")], "live")),
      "GET /api/chief/conversations/c-old/events?after=0&limit=200": () => page([ev(1, "parked")], "c-old"),
    });
    const { result, rerender } = renderHook(({ p }) => useJournal(p, "chat", true), { initialProps: { p: "/api/chief/events" } });
    rerender({ p: "/api/chief/conversations/c-old/events" });
    await waitFor(() => expect(result.current.events.map((e) => (e as { text: string }).text)).toEqual(["parked"]));
    release();
    await new Promise((r) => setTimeout(r, 30));
    expect(result.current.events.map((e) => (e as { text: string }).text)).toEqual(["parked"]);
  });

  it("shows nothing from the old path while the new one is loading", () => {
    mockApi({ "GET /api/chief/events?after=0&limit=200": () => page([ev(1, "live")], "live") });
    const { result, rerender } = renderHook(({ p }) => useJournal(p, "chat", true), { initialProps: { p: "/api/chief/events" } });
    rerender({ p: "/api/chief/conversations/c-x/events" });
    expect(result.current).toEqual({ events: [], error: null, loaded: false });
  });

  it("starts over when the server's log moves to another conversation, even if it is longer than ours", async () => {
    // Conversation B already has more entries than the cursor we hold from A: only its id tells them apart.
    const a = [ev(1, "a1"), ev(2, "a2"), ev(3, "a3")];
    const b = [ev(1, "b1"), ev(2, "b2"), ev(3, "b3"), ev(4, "b4"), ev(5, "b5")];
    let current = "a";
    mockApi({
      "GET /api/chief/events?after=0&limit=200": () => (current === "a" ? page(a, "a") : page(b, "b")),
      "GET /api/chief/events?after=3&limit=200": () => page(b.slice(3), "b"),
    });
    const { result } = renderHook(() => useJournal("/api/chief/events", "chat", true));
    await waitFor(() => expect(result.current.events.map((e) => (e as { text: string }).text)).toEqual(["a1", "a2", "a3"]));
    current = "b";
    act(() => FakeEventSource.emit({ type: "invalidate", resource: "chat" }));
    await waitFor(() => expect(result.current.events.map((e) => (e as { text: string }).text)).toEqual(["b1", "b2", "b3", "b4", "b5"]));
  });
});
