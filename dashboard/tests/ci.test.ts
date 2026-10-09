import { describe, it, expect, vi } from "vitest";
import { Hono } from "hono";
import { ciRoutes } from "../server/routes/ci";
import { GhError, type Gh } from "../server/lib/gh";
import type { CiRun } from "../shared/types";

const raw = (id: number, name: string, branch = "main") => ({
  databaseId: id, name, status: "completed", conclusion: "success",
  headBranch: branch, event: "push", createdAt: "2026-10-01T00:00:00Z", url: `https://example.test/runs/${id}`,
});

function setup(json: unknown) {
  const calls: string[][] = [];
  const gh: Gh = {
    async json<T>(args: string[]): Promise<T> {
      calls.push(args);
      return json as T;
    },
    text: async () => "",
  };
  const api = new Hono();
  ciRoutes(api, { gh });
  return { api, calls };
}

describe("GET /ci", () => {
  it("keeps only CI runs and maps field names", async () => {
    const { api, calls } = setup([raw(1, "CI", "phase-05"), raw(2, "Graph Update: pip in /."), raw(3, "CI")]);
    const r = await api.request("/ci");
    const body = (await r.json()) as { runs: CiRun[] };
    expect(body.runs.map((x) => x.id)).toEqual([1, 3]);
    expect(body.runs[0]).toEqual({
      id: 1, status: "completed", conclusion: "success", branch: "phase-05", event: "push",
      createdAt: "2026-10-01T00:00:00Z", url: "https://example.test/runs/1",
    });
    expect(calls[0]).toEqual([
      "run", "list", "--limit", "40", "--json", "databaseId,name,status,conclusion,headBranch,event,createdAt,url",
    ]);
  });
  it("returns at most 10 runs", async () => {
    const { api } = setup(Array.from({ length: 20 }, (_, i) => raw(i, "CI")));
    const body = (await (await api.request("/ci")).json()) as { runs: CiRun[] };
    expect(body.runs).toHaveLength(10);
  });
  it("502 with public message on GhError", async () => {
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    const gh: Gh = { json: async () => { throw new GhError("SECRET", [], 1); }, text: async () => "" };
    const api = new Hono();
    ciRoutes(api, { gh });
    const r = await api.request("/ci");
    expect(r.status).toBe(502);
    expect(await r.json()).toEqual({ error: "GitHub CLI failed" });
    spy.mockRestore();
  });
});
