import { describe, it, expect, vi } from "vitest";
import { Hono } from "hono";
import { issueRoutes } from "../server/routes/issues";
import { EventBus } from "../server/events";
import { GhError, type Gh } from "../server/lib/gh";
import type { DashEvent, Issue } from "../shared/types";

function setup(opts: { json?: unknown; text?: string | Error } = {}) {
  const jsonCalls: string[][] = [];
  const textCalls: string[][] = [];
  const gh: Gh = {
    async json<T>(args: string[]): Promise<T> {
      jsonCalls.push(args);
      return (opts.json ?? []) as T;
    },
    async text(args: string[]) {
      textCalls.push(args);
      if (opts.text instanceof Error) throw opts.text;
      return opts.text ?? "https://example.test/issues/9";
    },
  };
  const bus = new EventBus();
  const events: DashEvent[] = [];
  bus.subscribe((e) => events.push(e));
  const api = new Hono();
  issueRoutes(api, { gh, bus });
  return { api, jsonCalls, textCalls, events };
}
const post = (api: Hono, body: unknown, raw = false) =>
  api.request("/issues", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: raw ? (body as string) : JSON.stringify(body),
  });

describe("GET /issues", () => {
  it("drops task-labelled issues and maps labels to names", async () => {
    const { api, jsonCalls } = setup({
      json: [
        { number: 1, title: "a", labels: [{ name: "bug" }], createdAt: "c", updatedAt: "u", url: "x" },
        { number: 2, title: "b", labels: [{ name: "task" }, { name: "x" }], createdAt: "c", updatedAt: "u", url: "y" },
      ],
    });
    const r = await api.request("/issues");
    expect((await r.json()) as Issue[]).toEqual([
      { number: 1, title: "a", labels: ["bug"], createdAt: "c", updatedAt: "u", url: "x" },
    ]);
    expect(jsonCalls[0]).toEqual([
      "issue", "list", "--state", "open", "--limit", "100", "--json", "number,title,labels,createdAt,updatedAt,url",
    ]);
  });
  it("returns 502 with the public message on GhError", async () => {
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    const gh: Gh = {
      json: async () => {
        throw new GhError("SECRET-DETAIL", [], 1);
      },
      text: async () => "",
    };
    const api = new Hono();
    issueRoutes(api, { gh, bus: new EventBus() });
    const r = await api.request("/issues");
    expect(r.status).toBe(502);
    const t = await r.text();
    expect(t).not.toContain("SECRET-DETAIL");
    expect(JSON.parse(t)).toEqual({ error: "GitHub CLI failed" });
    spy.mockRestore();
  });
});

describe("POST /issues", () => {
  it("rejects invalid input with 400 and makes no gh call", async () => {
    const { api, textCalls } = setup();
    for (const b of [{ title: "" }, { title: "   " }, { title: "x".repeat(201) }, {}, { title: 5 }, { title: "ok", body: 3 }, []]) {
      expect((await post(api, b)).status).toBe(400);
    }
    expect((await post(api, "{not json", true)).status).toBe(400);
    expect(textCalls).toHaveLength(0);
  });
  it("passes title and body as separate argv elements, returns 201 and emits invalidate", async () => {
    const { api, textCalls, events } = setup();
    const title = '"; rm -rf /';
    const r = await post(api, { title: `  ${title}  `, body: "$(whoami)" });
    expect(r.status).toBe(201);
    expect(await r.json()).toEqual({ url: "https://example.test/issues/9" });
    expect(textCalls).toEqual([["issue", "create", "--title", title, "--body", "$(whoami)"]]);
    expect(events).toEqual([{ type: "invalidate", resource: "issues" }]);
  });
  it("defaults body to empty string", async () => {
    const { api, textCalls } = setup();
    await post(api, { title: "t" });
    expect(textCalls[0]).toEqual(["issue", "create", "--title", "t", "--body", ""]);
  });
  it("502 on gh failure, no event", async () => {
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    const { api, events } = setup({ text: new GhError("SECRET", [], 1) });
    const r = await post(api, { title: "t" });
    expect(r.status).toBe(502);
    expect(await r.json()).toEqual({ error: "GitHub CLI failed" });
    expect(events).toHaveLength(0);
    spy.mockRestore();
  });
});
