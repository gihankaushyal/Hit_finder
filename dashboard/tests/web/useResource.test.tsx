import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, renderHook, waitFor } from "@testing-library/react";
import { useResource } from "../../web/src/lib/useResource";
import { RECONNECT_MIN_MS } from "../../web/src/lib/events";
import { UNAUTHORIZED_MESSAGE } from "../../web/src/lib/api";

class FakeEventSource {
  static instances: FakeEventSource[] = [];
  readyState = 0;
  onmessage: ((e: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  onopen: (() => void) | null = null;
  closed = false;
  constructor(public url: string) {
    FakeEventSource.instances.push(this);
  }
  close() {
    this.closed = true;
    this.readyState = 2;
  }
  open() {
    this.readyState = 1;
    this.onopen?.();
  }
  send(data: unknown) {
    this.onmessage?.({ data: JSON.stringify(data) });
  }
}
const live = () => FakeEventSource.instances.filter((i) => !i.closed);
const res = (status: number, body: unknown) => ({ ok: status >= 200 && status < 300, status, json: async () => body });

let fetchMock: ReturnType<typeof vi.fn>;
beforeEach(() => {
  FakeEventSource.instances = [];
  fetchMock = vi.fn().mockResolvedValue(res(200, [{ n: 1 }]));
  vi.stubGlobal("fetch", fetchMock);
  vi.stubGlobal("EventSource", FakeEventSource);
});
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("useResource", () => {
  it("starts loading, then returns the data", async () => {
    const { result } = renderHook(() => useResource<{ n: number }[]>("issues", "/api/issues"));
    expect(result.current.loading).toBe(true);
    expect(result.current.data).toBeNull();
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.data).toEqual([{ n: 1 }]);
    expect(result.current.error).toBeNull();
    expect(result.current.empty).toBe(false);
    expect(fetchMock).toHaveBeenCalledWith("/api/issues", expect.objectContaining({ credentials: "same-origin" }));
  });

  it("an empty array is reported as empty, distinct from loading and error", async () => {
    fetchMock.mockResolvedValue(res(200, []));
    const { result } = renderHook(() => useResource<unknown[]>("issues", "/api/issues"));
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.empty).toBe(true);
    expect(result.current.error).toBeNull();
  });

  it("a failed fetch sets error and is not loading or empty", async () => {
    fetchMock.mockResolvedValue(res(502, { error: "GitHub CLI failed" }));
    const { result } = renderHook(() => useResource("issues", "/api/issues"));
    await waitFor(() => expect(result.current.error).toBe("GitHub CLI failed"));
    expect(result.current.loading).toBe(false);
    expect(result.current.empty).toBe(false);
    expect(result.current.unauthorized).toBe(false);
  });

  it("a 401 is flagged unauthorized with the open-the-tokenised-URL message", async () => {
    fetchMock.mockResolvedValue(res(401, { error: "unauthorized" }));
    const { result } = renderHook(() => useResource("issues", "/api/issues"));
    await waitFor(() => expect(result.current.unauthorized).toBe(true));
    expect(result.current.error).toBe(UNAUTHORIZED_MESSAGE);
  });

  it("an invalidate event for the same resource refetches; one for another resource does not", async () => {
    const { result } = renderHook(() => useResource("issues", "/api/issues"));
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(fetchMock).toHaveBeenCalledTimes(1);
    act(() => FakeEventSource.instances[0].send({ type: "invalidate", resource: "prs" }));
    act(() => FakeEventSource.instances[0].send({ type: "tests-line", line: "x" }));
    await act(async () => {});
    expect(fetchMock).toHaveBeenCalledTimes(1);
    act(() => FakeEventSource.instances[0].send({ type: "invalidate", resource: "issues" }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
  });

  it("malformed stream data is ignored", async () => {
    const { result } = renderHook(() => useResource("issues", "/api/issues"));
    await waitFor(() => expect(result.current.loading).toBe(false));
    act(() => FakeEventSource.instances[0].onmessage?.({ data: "{not json" }));
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("keeps the previous data visible while reloading and after a failed reload", async () => {
    const { result } = renderHook(() => useResource<{ n: number }[]>("issues", "/api/issues"));
    await waitFor(() => expect(result.current.data).toEqual([{ n: 1 }]));
    let release: (v: unknown) => void = () => {};
    fetchMock.mockReturnValueOnce(new Promise((r) => (release = r)));
    act(() => result.current.reload());
    expect(result.current.data).toEqual([{ n: 1 }]);
    expect(result.current.loading).toBe(false);
    await act(async () => release(res(500, { error: "boom" })));
    expect(result.current.data).toEqual([{ n: 1 }]);
    expect(result.current.error).toBe("boom");
  });

  it("shares one EventSource between hooks and closes it, with no leftover listener, when the last one unmounts", async () => {
    const a = renderHook(() => useResource("issues", "/api/issues"));
    const b = renderHook(() => useResource("prs", "/api/prs"));
    await waitFor(() => expect(a.result.current.loading).toBe(false));
    expect(FakeEventSource.instances).toHaveLength(1);
    a.unmount();
    expect(live()).toHaveLength(1);
    const before = fetchMock.mock.calls.length;
    act(() => FakeEventSource.instances[0].send({ type: "invalidate", resource: "issues" }));
    await act(async () => {});
    expect(fetchMock.mock.calls.length).toBe(before); // the unmounted hook no longer reacts
    b.unmount();
    expect(live()).toHaveLength(0);
  });

  it("does not update state after unmount", async () => {
    let release: (v: unknown) => void = () => {};
    fetchMock.mockReturnValueOnce(new Promise((r) => (release = r)));
    const err = vi.spyOn(console, "error").mockImplementation(() => {});
    const { unmount } = renderHook(() => useResource("issues", "/api/issues"));
    unmount();
    await act(async () => release(res(200, [])));
    expect(err).not.toHaveBeenCalled();
    err.mockRestore();
  });

  it("reconnects after the stream drops for good, and reloads to catch what it missed", async () => {
    vi.useFakeTimers();
    const { result } = renderHook(() => useResource("issues", "/api/issues"));
    await act(async () => {});
    expect(result.current.loading).toBe(false);
    const first = FakeEventSource.instances[0];
    first.open();
    first.readyState = 2; // the browser gave up (for example after a 401 or a server restart)
    act(() => first.onerror?.());
    expect(first.closed).toBe(true);
    expect(FakeEventSource.instances).toHaveLength(1);
    act(() => {
      vi.advanceTimersByTime(RECONNECT_MIN_MS);
    });
    expect(FakeEventSource.instances).toHaveLength(2);
    const calls = fetchMock.mock.calls.length;
    await act(async () => FakeEventSource.instances[1].open());
    expect(fetchMock.mock.calls.length).toBeGreaterThan(calls);
  });

  it("a native reconnect (the browser retries by itself) also reloads on the next open", async () => {
    const { result } = renderHook(() => useResource("issues", "/api/issues"));
    await waitFor(() => expect(result.current.loading).toBe(false));
    const es = FakeEventSource.instances[0];
    es.open();
    const calls = fetchMock.mock.calls.length;
    es.readyState = 0; // CONNECTING: the browser is retrying
    act(() => es.onerror?.());
    expect(FakeEventSource.instances).toHaveLength(1);
    await act(async () => es.open());
    await waitFor(() => expect(fetchMock.mock.calls.length).toBeGreaterThan(calls));
  });

  it("a pending reconnect timer is cleared on unmount", async () => {
    vi.useFakeTimers();
    const { unmount } = renderHook(() => useResource("issues", "/api/issues"));
    await act(async () => {});
    const first = FakeEventSource.instances[0];
    first.readyState = 2;
    act(() => first.onerror?.());
    unmount();
    act(() => {
      vi.advanceTimersByTime(RECONNECT_MIN_MS * 20);
    });
    expect(FakeEventSource.instances).toHaveLength(1);
  });
});
