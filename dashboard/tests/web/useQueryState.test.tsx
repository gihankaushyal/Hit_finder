import { describe, it, expect } from "vitest";
import { act, renderHook } from "@testing-library/react";
import { setQueryParams, tabHref, useQueryParam } from "../../web/src/lib/useQueryState";

describe("useQueryParam", () => {
  it("reads the fallback when the param is absent and the value when present", () => {
    window.history.replaceState(null, "", "/?tab=kanban");
    expect(renderHook(() => useQueryParam("tab", "overview")).result.current[0]).toBe("kanban");
    expect(renderHook(() => useQueryParam("kind", "all")).result.current[0]).toBe("all");
  });

  it("writes the value to the address and re-renders", () => {
    const { result } = renderHook(() => useQueryParam("kind", "all"));
    act(() => result.current[1]("ssl-pretrain"));
    expect(result.current[0]).toBe("ssl-pretrain");
    expect(window.location.search).toBe("?kind=ssl-pretrain");
  });

  it("keeps the default value out of the address", () => {
    window.history.replaceState(null, "", "/?kind=legacy&dir=desc");
    const { result } = renderHook(() => useQueryParam("kind", "all"));
    act(() => result.current[1]("all"));
    expect(window.location.search).toBe("?dir=desc");
  });

  it("two hooks on the same param stay in step", () => {
    const a = renderHook(() => useQueryParam("tab", "overview"));
    const b = renderHook(() => useQueryParam("tab", "overview"));
    act(() => a.result.current[1]("runs"));
    expect(b.result.current[0]).toBe("runs");
  });

  it("push adds a history entry that Back undoes", async () => {
    const start = window.history.length;
    const { result } = renderHook(() => useQueryParam("tab", "overview"));
    act(() => result.current[1]("runs", "push"));
    expect(window.history.length).toBe(start + 1);
    await act(async () => {
      window.history.back();
      await new Promise((r) => setTimeout(r, 30));
    });
    expect(result.current[0]).toBe("overview");
  });

  it("setQueryParams changes several params at once and drops nulls", () => {
    window.history.replaceState(null, "", "/?a=1&b=2");
    act(() => setQueryParams({ a: null, c: "3" }));
    expect(window.location.search).toBe("?b=2&c=3");
  });

describe("tabHref", () => {
  it("carries the other params into a tab address", () => {
    expect(tabHref("?tab=runs&kind=track1&sort=crossAp&dir=desc", "kanban", "overview")).toBe("?tab=kanban&kind=track1&sort=crossAp&dir=desc");
  });
  it("leaves the default tab out of the address, and is just ? when nothing else is set", () => {
    expect(tabHref("?tab=runs&kind=track1", "overview", "overview")).toBe("?kind=track1");
    expect(tabHref("?tab=runs", "overview", "overview")).toBe("?");
  });
});
});
