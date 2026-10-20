import { describe, it, expect } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { assertSafeEnv, DASHBOARD_DIR, FAKE_GH, FAKE_PYTEST } from "../e2e/harness";

describe("e2e harness guard", () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "dash-guard-"));
  const good = (): Record<string, string> => ({
    DASH_KANBAN: path.join(tmp, "repo", "phase-05-kanban.md"),
    DASH_STATE_DIR: path.join(tmp, "state"),
    DASH_REPO_ROOT: path.join(tmp, "repo"),
    DASH_CHECKPOINTS: path.join(tmp, "repo", "checkpoints"),
    FAKE_GH_STATE: path.join(tmp, "gh.json"),
    FAKE_GH_CALLS: path.join(tmp, "calls"),
    DASH_GH_BIN: FAKE_GH,
    DASH_PYTHON: FAKE_PYTEST,
    DASH_AGENTS_DIR: path.join(tmp, "agents"),
    DASH_AGENT_DRIVER: "fake",
    DASH_AGENT_BASE: "e2e-branch",
  });

  it("accepts a configuration entirely inside the temp directory with the fakes", () => {
    expect(() => assertSafeEnv(good(), tmp)).not.toThrow();
  });
  it.each(["DASH_KANBAN", "DASH_STATE_DIR", "DASH_REPO_ROOT", "DASH_CHECKPOINTS", "FAKE_GH_STATE", "FAKE_GH_CALLS"])("refuses %s outside the temp directory", (key) => {
    expect(() => assertSafeEnv({ ...good(), [key]: path.join(DASHBOARD_DIR, "..", "elsewhere") }, tmp)).toThrow(/not inside the temp directory/);
  });
  it("refuses the real kanban file and the real state directory", () => {
    expect(() => assertSafeEnv({ ...good(), DASH_KANBAN: path.resolve(DASHBOARD_DIR, "..", "phase-05-kanban.md") }, tmp)).toThrow();
    expect(() => assertSafeEnv({ ...good(), DASH_STATE_DIR: path.join(DASHBOARD_DIR, ".state") }, tmp)).toThrow();
  });
  it("refuses a missing variable", () => {
    const env = good();
    delete env.DASH_KANBAN;
    expect(() => assertSafeEnv(env, tmp)).toThrow(/DASH_KANBAN is not set/);
  });
  it("treats a whitespace-only path as missing, the way the server does, instead of resolving it against the cwd", () => {
    for (const key of ["DASH_KANBAN", "DASH_STATE_DIR", "DASH_REPO_ROOT", "DASH_CHECKPOINTS"]) {
      expect(() => assertSafeEnv({ ...good(), [key]: "   " }, tmp)).toThrow(new RegExp(`${key} is not set`));
    }
  });
  it("refuses the real gh or python", () => {
    expect(() => assertSafeEnv({ ...good(), DASH_GH_BIN: "gh" }, tmp)).toThrow(/DASH_GH_BIN must be/);
    expect(() => assertSafeEnv({ ...good(), DASH_GH_BIN: "/usr/bin/gh" }, tmp)).toThrow(/DASH_GH_BIN must be/);
    expect(() => assertSafeEnv({ ...good(), DASH_PYTHON: "python" }, tmp)).toThrow(/DASH_PYTHON must be/);
  });
  it("refuses an agents directory outside the temp directory", () => {
    expect(() => assertSafeEnv({ ...good(), DASH_AGENTS_DIR: path.join(DASHBOARD_DIR, "..", "..", "Hit_finder-agents") }, tmp)).toThrow(/DASH_AGENTS_DIR=.*not inside the temp directory/);
  });
  it("refuses anything but the fake agent driver, so no real Claude session can start", () => {
    for (const driver of ["sdk", "", undefined]) {
      expect(() => assertSafeEnv({ ...good(), DASH_AGENT_DRIVER: driver as string }, tmp)).toThrow(/DASH_AGENT_DRIVER must be "fake"/);
    }
  });
  it("refuses a claude binary override", () => {
    expect(() => assertSafeEnv({ ...good(), DASH_CLAUDE_BIN: "/usr/local/bin/claude" }, tmp)).toThrow(/DASH_CLAUDE_BIN must not be set/);
  });
  it("refuses a temp directory inside the dashboard directory", () => {
    expect(() => assertSafeEnv(good(), path.join(DASHBOARD_DIR, "tmp-run"))).toThrow(/must not be inside the dashboard/);
  });
});
