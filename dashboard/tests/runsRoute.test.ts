import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { Hono } from "hono";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { runRoutes } from "../server/routes/runs";
import { createRunScanner } from "../server/lib/runScan";
import { buildCheckpoints } from "./helpers/runFixtures";

let tmp: string;
beforeEach(() => { tmp = fs.mkdtempSync(path.join(os.tmpdir(), "runsroute-")); });
afterEach(() => fs.rmSync(tmp, { recursive: true, force: true }));

function appFor(scanner: { scan(): unknown }) {
  const app = new Hono();
  const api = new Hono();
  runRoutes(api, { scanner: scanner as never });
  app.route("/api", api);
  return app;
}

describe("GET /api/runs", () => {
  it("returns the scan as JSON", async () => {
    const res = await appFor(createRunScanner(buildCheckpoints(tmp))).request("/api/runs");
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.runs.length).toBe(9);
    expect(body.checkpointsDirExists).toBe(true);
    expect(JSON.stringify(body)).not.toContain(tmp);
  });
  it("an unexpected failure is a 500 with a fixed message", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    const res = await appFor({ scan: () => { throw new Error("secret /home/x"); } }).request("/api/runs");
    expect(res.status).toBe(500);
    expect(await res.json()).toEqual({ error: "internal error" });
  });
});
