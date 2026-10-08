import { describe, it, expect, vi } from "vitest";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Hono } from "hono";
import { prRoutes } from "../server/routes/prs";
import { GhError, type Gh } from "../server/lib/gh";
import type { PrsResponse } from "../shared/types";

const fixtures = path.join(path.dirname(fileURLToPath(import.meta.url)), "fixtures");
const list = JSON.parse(fs.readFileSync(path.join(fixtures, "pr-list.json"), "utf8"));
const view = JSON.parse(fs.readFileSync(path.join(fixtures, "pr-view-47.json"), "utf8"));

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

  describe("check state of open PRs", () => {
    const run = (status: string, conclusion: string) => ({ __typename: "CheckRun", name: "x", status, conclusion });
    const ctx = (state: string) => ({ __typename: "StatusContext", context: "y", state });
    async function checksFor(rollup: unknown, withKey = true) {
      const open = { ...list.find((p: { state: string }) => p.state === "OPEN"), ...(withKey ? { statusCheckRollup: rollup } : {}) };
      const rest = list.filter((p: { state: string }) => p.state !== "OPEN");
      const calls: string[][] = [];
      const gh: Gh = {
        async json<T>(args: string[]): Promise<T> {
          calls.push(args);
          return (args[1] === "list" ? [...rest, open] : view) as T;
        },
        text: async () => "",
      };
      const body = (await (await mk(gh).request("/prs")).json()) as PrsResponse;
      return { checks: body.open[0].checks, open: body.open[0], calls };
    }

    it("asks gh for statusCheckRollup on the list call", async () => {
      const { calls } = await checksFor([]);
      expect(calls[0].join(" ")).toContain("statusCheckRollup");
    });
    it.each([
      ["no checks", [], { state: "none", passing: 0, failing: 0, pending: 0 }],
      ["missing key", undefined, { state: "none", passing: 0, failing: 0, pending: 0 }],
      ["all green", [run("COMPLETED", "SUCCESS"), ctx("SUCCESS"), run("COMPLETED", "SKIPPED"), run("COMPLETED", "NEUTRAL")], { state: "passing", passing: 4, failing: 0, pending: 0 }],
      ["one failing wins over pending", [run("COMPLETED", "SUCCESS"), run("IN_PROGRESS", ""), run("COMPLETED", "FAILURE")], { state: "failing", passing: 1, failing: 1, pending: 1 }],
      ["pending only", [run("COMPLETED", "SUCCESS"), run("QUEUED", ""), ctx("PENDING")], { state: "pending", passing: 1, failing: 0, pending: 2 }],
      ["timed out, cancelled and error count as failing", [run("COMPLETED", "TIMED_OUT"), run("COMPLETED", "CANCELLED"), ctx("ERROR")], { state: "failing", passing: 0, failing: 3, pending: 0 }],
    ])("%s", async (_name, rollup, expected) => {
      const { checks, open } = await checksFor(rollup, rollup !== undefined);
      expect(checks).toEqual(expected);
      expect(open).not.toHaveProperty("statusCheckRollup");
    });
    it("merged PRs carry no check state", async () => {
      const body = (await (await mk(fakeGh()).request("/prs")).json()) as PrsResponse;
      expect(body.recent[0]).not.toHaveProperty("checks");
      expect(body.latest).not.toHaveProperty("statusCheckRollup");
    });
  });
});
