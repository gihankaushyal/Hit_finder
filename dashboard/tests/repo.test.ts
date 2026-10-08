import { describe, it, expect, vi } from "vitest";
import { Hono } from "hono";
import { repoRoutes } from "../server/routes/repo";
import type { Runner } from "../server/runner";

function setup(fn: (args: string[]) => { stdout?: string; stderr?: string; code: number }) {
  const calls: { cmd: string; args: string[]; cwd?: string }[] = [];
  const runner: Runner = async (cmd, args, opts) => {
    calls.push({ cmd, args, cwd: opts?.cwd });
    const r = fn(args);
    return { stdout: r.stdout ?? "", stderr: r.stderr ?? "", code: r.code };
  };
  const api = new Hono();
  repoRoutes(api, { runner, repoRoot: "/some/repo" });
  return { api, calls };
}

describe("GET /repo", () => {
  it("returns the branch and short commit from git, run in the repo root with argument arrays", async () => {
    const { api, calls } = setup((args) =>
      args[0] === "rev-parse" && args.includes("--abbrev-ref")
        ? { stdout: "phase-05-dashboard-status\n", code: 0 }
        : { stdout: "98ad636\n", code: 0 },
    );
    const res = await api.request("/repo");
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ branch: "phase-05-dashboard-status", commit: "98ad636" });
    expect(calls.map((c) => [c.cmd, ...c.args])).toEqual([
      ["git", "rev-parse", "--abbrev-ref", "HEAD"],
      ["git", "rev-parse", "--short", "HEAD"],
    ]);
    expect(calls.every((c) => c.cwd === "/some/repo")).toBe(true);
  });

  it("a git failure is a fixed error message with no raw stderr", async () => {
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    const { api } = setup(() => ({ code: 128, stderr: "fatal: not a git repository: /secret/path" }));
    const res = await api.request("/repo");
    spy.mockRestore();
    expect(res.status).toBe(500);
    const text = await res.text();
    expect(text).not.toContain("secret");
    expect(JSON.parse(text)).toEqual({ error: "internal error" });
  });
});
