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
    FAKE_GH_STATE: path.join(tmp, "gh.json"),
    FAKE_GH_CALLS: path.join(tmp, "calls"),
    DASH_GH_BIN: FAKE_GH,
    DASH_PYTHON: FAKE_PYTEST,
  });

  it("accepts a configuration entirely inside the temp directory with the fakes", () => {
    expect(() => assertSafeEnv(good(), tmp)).not.toThrow();
  });
  it.each(["DASH_KANBAN", "DASH_STATE_DIR", "DASH_REPO_ROOT", "FAKE_GH_STATE", "FAKE_GH_CALLS"])("refuses %s outside the temp directory", (key) => {
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
  it("refuses the real gh or python", () => {
    expect(() => assertSafeEnv({ ...good(), DASH_GH_BIN: "gh" }, tmp)).toThrow(/DASH_GH_BIN must be/);
    expect(() => assertSafeEnv({ ...good(), DASH_GH_BIN: "/usr/bin/gh" }, tmp)).toThrow(/DASH_GH_BIN must be/);
    expect(() => assertSafeEnv({ ...good(), DASH_PYTHON: "python" }, tmp)).toThrow(/DASH_PYTHON must be/);
  });
  it("refuses a temp directory inside the dashboard directory", () => {
    expect(() => assertSafeEnv(good(), path.join(DASHBOARD_DIR, "tmp-run"))).toThrow(/must not be inside the dashboard/);
  });
});
