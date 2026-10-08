import { describe, it, expect, vi } from "vitest";
import fs from "node:fs";
import { Hono } from "hono";
import { prRoutes } from "../server/routes/prs";
import { GhError, type Gh } from "../server/lib/gh";
import type { PrsResponse } from "../shared/types";

const list = JSON.parse(fs.readFileSync("tests/fixtures/pr-list.json", "utf8"));
const view = JSON.parse(fs.readFileSync("tests/fixtures/pr-view-47.json", "utf8"));

function fakeGh(calls: string[][] = []): Gh {
  return {
    async json<T>(args: string[]): Promise<T> {
      calls.push(args);
      if (args[1] === "list") return list as T;
      if (args[1] === "view") return view as T;
      throw new Error("unexpected " + args.join(" "));
    },
    async text() {
      return "";
    },
  };
}
function mk(gh: Gh): Hono {
  const api = new Hono();
  prRoutes(api, { gh });
  return api;
}

describe("GET /prs", () => {
  it("returns latest merged with detail, four recent, and open PRs", async () => {
    const calls: string[][] = [];
    const r = await mk(fakeGh(calls)).request("/prs");
    expect(r.status).toBe(200);
    const body = (await r.json()) as PrsResponse;
    expect(body.latest?.number).toBe(47);
    expect(body.latest!.summary.length).toBeGreaterThan(0);
    expect(body.latest!.testPlan).toHaveLength(2);
    expect(body.latest!.files).toHaveLength(2);
    expect(body.recent.map((p) => p.number)).toEqual([46, 45, 44, 43]);
    expect(body.open.map((p) => p.number)).toEqual([50]);
    expect(calls[1]).toEqual(expect.arrayContaining(["pr", "view", "47"]));
    expect(body.latest).not.toHaveProperty("body");
  });
  it("handles no merged PRs", async () => {
    const gh: Gh = { json: async <T,>() => [] as T, text: async () => "" };
    const body = (await (await mk(gh).request("/prs")).json()) as PrsResponse;
    expect(body).toEqual({ latest: null, recent: [], open: [] });
  });
  it("returns 502 with only the public message on GhError", async () => {
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    const gh: Gh = {
      json: async () => {
        throw new GhError("gh pr list failed: SECRET-DETAIL", ["pr", "list"], 1);
      },
      text: async () => "",
    };
    const r = await mk(gh).request("/prs");
    expect(r.status).toBe(502);
    const text = await r.text();
    expect(JSON.parse(text)).toEqual({ error: "GitHub CLI failed" });
    expect(text).not.toContain("SECRET-DETAIL");
    expect(spy).toHaveBeenCalled();
    spy.mockRestore();
  });
});
