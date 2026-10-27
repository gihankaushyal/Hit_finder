import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { subscribe } from "../../web/src/lib/events";
import { FakeEventSource, stubEventSource } from "./helpers";

beforeEach(() => stubEventSource());
afterEach(() => vi.unstubAllGlobals());

describe("events.subscribe", () => {
  it("passes agent-state events to onAgentState and not to reload", () => {
    const onAgentState = vi.fn();
    const reload = vi.fn();
    const stop = subscribe({ resource: "agents", reload, onAgentState });
    FakeEventSource.emit({ type: "agent-state", id: "a", name: "a", state: "failed", error: "x" });
    expect(onAgentState).toHaveBeenCalledWith({ type: "agent-state", id: "a", name: "a", state: "failed", error: "x" });
    expect(reload).not.toHaveBeenCalled();
    expect(FakeEventSource.live).toHaveLength(1); // still one shared EventSource
    stop();
  });
});
