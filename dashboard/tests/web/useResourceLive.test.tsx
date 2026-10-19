import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, render, renderHook, screen } from "@testing-library/react";
import { useResource, CI_REFRESH_MS, PRS_REFRESH_MS, REPO_REFRESH_MS } from "../../web/src/lib/useResource";
import { TopBar } from "../../web/src/panels/TopBar";
import { TestStatus } from "../../web/src/panels/TestStatus";
import { LatestPr } from "../../web/src/panels/LatestPr";
import { FakeEventSource, mockApi, stubEventSource } from "./helpers";

const res = (status: number, body: unknown) => ({ ok: status >= 200 && status < 300, status, json: async () => body });
const flush = () => act(async () => {});
const tick = (ms: number) => act(async () => void (await vi.advanceTimersByTimeAsync(ms)));
const setVisibility = (state: "visible" | "hidden") => {
  Object.defineProperty(document, "visibilityState", { value: state, configurable: true });
  act(() => void document.dispatchEvent(new Event("visibilitychange")));
};

let fetchMock: ReturnType<typeof vi.fn>;
beforeEach(() => {
  vi.useFakeTimers();
  stubEventSource();
  fetchMock = vi.fn().mockResolvedValue(res(200, { n: 1 }));
  vi.stubGlobal("fetch", fetchMock);
});
afterEach(() => {
  setVisibility("visible");
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("useResource polling", () => {
  it("refetches every refreshMs when asked to", async () => {
    renderHook(() => useResource("ci", "/api/ci", undefined, 1000));
    await flush();
    expect(fetchMock).toHaveBeenCalledTimes(1);
    await tick(1000);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    await tick(2000);
    expect(fetchMock).toHaveBeenCalledTimes(4);
  });

  it("does not poll without refreshMs", async () => {
    renderHook(() => useResource("issues", "/api/issues"));
    await flush();
    await tick(10 * 60_000);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("pauses while the tab is hidden and refetches immediately when it becomes visible", async () => {
    renderHook(() => useResource("ci", "/api/ci", undefined, 1000));
    await flush();
    setVisibility("hidden");
    await tick(10_000);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    setVisibility("visible");
    await flush();
    expect(fetchMock).toHaveBeenCalledTimes(2);
    await tick(1000);
    expect(fetchMock).toHaveBeenCalledTimes(3);
  });

  it("clears its timer and listener on unmount", async () => {
    const { unmount } = renderHook(() => useResource("ci", "/api/ci", undefined, 1000));
    await flush();
    unmount();
    expect(vi.getTimerCount()).toBe(0);
    await tick(10_000);
    setVisibility("hidden");
    setVisibility("visible");
    await flush();
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("a polled refetch keeps showing the data instead of the loading skeleton", async () => {
    const { result } = renderHook(() => useResource<{ n: number }>("ci", "/api/ci", undefined, 1000));
    await flush();
    expect(result.current.data).toEqual({ n: 1 });
    let release: (v: unknown) => void = () => {};
    fetchMock.mockReturnValueOnce(new Promise((r) => (release = r)));
    await tick(1000);
    expect(result.current.loading).toBe(false);
    expect(result.current.data).toEqual({ n: 1 });
    await act(async () => release(res(200, { n: 2 })));
    expect(result.current.data).toEqual({ n: 2 });
    expect(result.current.loading).toBe(false);
  });
});

describe("useResource shared store", () => {
  it("two hooks on one path share one request and update together", async () => {
    const a = renderHook(() => useResource<{ n: number }>("ci", "/api/ci"));
    const b = renderHook(() => useResource<{ n: number }>("ci", "/api/ci"));
    await flush();
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(a.result.current.data).toEqual({ n: 1 });
    expect(b.result.current.data).toEqual({ n: 1 });
    fetchMock.mockResolvedValue(res(200, { n: 2 }));
    act(() => FakeEventSource.emit({ type: "invalidate", resource: "ci" }));
    await flush();
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(a.result.current.data).toEqual({ n: 2 });
    expect(b.result.current.data).toEqual({ n: 2 });
    act(() => a.result.current.reload());
    await flush();
    expect(b.result.current.data).toEqual({ n: 2 });
    expect(fetchMock).toHaveBeenCalledTimes(3);
  });

  it("a hook mounting later reuses the cached value without flashing loading", async () => {
    renderHook(() => useResource("ci", "/api/ci"));
    await flush();
    const late = renderHook(() => useResource<{ n: number }>("ci", "/api/ci"));
    expect(late.result.current.loading).toBe(false);
    expect(late.result.current.data).toEqual({ n: 1 });
  });

  it("different paths are independent, and each keeps its own empty rule", async () => {
    fetchMock.mockResolvedValue(res(200, []));
    const a = renderHook(() => useResource<unknown[]>("issues", "/api/issues"));
    const b = renderHook(() => useResource<unknown[]>("prs", "/api/prs", () => false));
    await flush();
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(a.result.current.empty).toBe(true);
    expect(b.result.current.empty).toBe(false);
  });

  it("releases the entry when the last subscriber unmounts: a remount fetches afresh", async () => {
    const a = renderHook(() => useResource("ci", "/api/ci"));
    const b = renderHook(() => useResource("ci", "/api/ci"));
    await flush();
    a.unmount();
    expect(FakeEventSource.live).toHaveLength(1); // still wanted by b
    b.unmount();
    expect(FakeEventSource.live).toHaveLength(0);
    const c = renderHook(() => useResource("ci", "/api/ci"));
    expect(c.result.current.loading).toBe(true);
    await flush();
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("TopBar and TestStatus together fetch /api/ci and /api/tests once each", async () => {
    const api = mockApi({
      "GET /api/ci": { body: { runs: [] } },
      "GET /api/tests": { body: { status: "idle", startedAt: null, finishedAt: null, commit: null, summary: null, tail: [], error: null } },
      "GET /api/repo": { body: { branch: "b", commit: "c" } },
    });
    render(
      <>
        <TopBar />
        <TestStatus />
      </>,
    );
    await flush();
    expect(api.count("GET", "/api/ci")).toBe(1);
    expect(api.count("GET", "/api/tests")).toBe(1);
  });
});

describe("panel refresh wiring", () => {
  it("repo polls every 30 s, ci and prs every 60 s", async () => {
    expect([REPO_REFRESH_MS, CI_REFRESH_MS, PRS_REFRESH_MS]).toEqual([30_000, 60_000, 60_000]);
    const api = mockApi({
      "GET /api/ci": { body: { runs: [] } },
      "GET /api/tests": { body: { status: "idle", startedAt: null, finishedAt: null, commit: null, summary: null, tail: [], error: null } },
      "GET /api/repo": { body: { branch: "b", commit: "c" } },
      "GET /api/prs": { body: { open: [], recent: [], latest: null } },
    });
    render(
      <>
        <TopBar />
        <TestStatus />
        <LatestPr />
      </>,
    );
    await flush();
    await tick(30_000);
    expect(api.count("GET", "/api/repo")).toBe(2);
    expect(api.count("GET", "/api/ci")).toBe(1);
    await tick(30_000);
    expect(api.count("GET", "/api/ci")).toBe(2);
    expect(api.count("GET", "/api/prs")).toBe(2);
    expect(api.count("GET", "/api/repo")).toBe(3);
    expect(api.count("GET", "/api/tests")).toBe(1);
    expect(screen.getByText("Hit_finder")).toBeInTheDocument();
  });
});
