import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { clearToasts, dismissToast, getToasts, notifyFinished, outcomeText, pushToast, setTabAttention, TOAST_MS } from "../../web/src/lib/attentionEffects";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  vi.useRealTimers();
  clearToasts();
  document.title = "";
  document.head.querySelectorAll("link[rel='icon']").forEach((l) => l.remove());
});

describe("setTabAttention", () => {
  beforeEach(() => {
    document.title = "Hit_finder";
  });
  it("prefixes the title with the count and restores it", () => {
    setTabAttention(2, "red");
    expect(document.title).toBe("(2) Hit_finder");
    setTabAttention(3, "green");
    expect(document.title).toBe("(3) Hit_finder");
    setTabAttention(0, null);
    expect(document.title).toBe("Hit_finder");
  });
  it("leaves the head untouched when there is no dot to draw or restore", () => {
    setTabAttention(0, null);
    expect(document.head.querySelector("link[rel~='icon']")).toBeNull();
  });
  it("does not throw without a canvas (jsdom)", () => {
    expect(() => setTabAttention(1, "red")).not.toThrow();
    setTabAttention(0, null);
  });
});

describe("notifyFinished", () => {
  it("does nothing when permission is not granted", () => {
    const Ctor = vi.fn();
    vi.stubGlobal("Notification", Object.assign(Ctor, { permission: "default" }));
    notifyFinished("a", "done");
    expect(Ctor).not.toHaveBeenCalled();
  });
  it("notifies only when granted and the tab is hidden", () => {
    const Ctor = vi.fn();
    vi.stubGlobal("Notification", Object.assign(Ctor, { permission: "granted" }));
    const hidden = vi.spyOn(document, "hidden", "get").mockReturnValue(false);
    notifyFinished("a", "done");
    expect(Ctor).not.toHaveBeenCalled();
    hidden.mockReturnValue(true);
    notifyFinished("a", "failed");
    expect(Ctor).toHaveBeenCalledWith("a failed", expect.objectContaining({ tag: "agent-a" }));
  });
  it("swallows a throwing Notification constructor", () => {
    vi.stubGlobal("Notification", Object.assign(function Boom() { throw new Error("x"); }, { permission: "granted" }));
    vi.spyOn(document, "hidden", "get").mockReturnValue(true);
    expect(() => notifyFinished("a", "done")).not.toThrow();
  });
});

describe("toasts", () => {
  it("adds a toast and removes it after TOAST_MS", () => {
    vi.useFakeTimers();
    pushToast("a finished", "green");
    expect(getToasts().map((t) => t.text)).toEqual(["a finished"]);
    vi.advanceTimersByTime(TOAST_MS + 1);
    expect(getToasts()).toEqual([]);
  });
  it("can be dismissed by id", () => {
    pushToast("x", "red");
    dismissToast(getToasts()[0].id);
    expect(getToasts()).toEqual([]);
  });
  it("outcomeText names the agent and the outcome", () => {
    expect(outcomeText("a", "failed")).toBe("a failed");
    expect(outcomeText("a", "done")).toBe("a finished");
    expect(outcomeText("a", "idle")).toBe("a finished a turn");
  });
});
