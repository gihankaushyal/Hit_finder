import { describe, it, expect } from "vitest";
import path from "node:path";
import { loadConfig } from "../server/config";
describe("loadConfig", () => {
  it("uses defaults", () => {
    const c = loadConfig({});
    expect(c.host).toBe("127.0.0.1");
    expect(c.port).toBe(4317);
    expect(c.ghBin).toBe("gh");
    expect(path.basename(c.kanbanPath)).toBe("phase-05-kanban.md");
    expect(c.kanbanPath.startsWith(c.repoRoot)).toBe(true);
    expect(path.basename(c.stateDir)).toBe(".state");
  });
  it("honours env overrides", () => {
    const c = loadConfig({ DASH_PORT: "5000", DASH_REPO_ROOT: "/r", DASH_GH_BIN: "/x/gh", DASH_KANBAN: "/k.md" });
    expect(c.port).toBe(5000);
    expect(c.repoRoot).toBe("/r");
    expect(c.ghBin).toBe("/x/gh");
    expect(c.kanbanPath).toBe("/k.md");
  });
  it("resolves relative paths to absolute ones", () => {
    const c = loadConfig({ DASH_STATE_DIR: "rel/state", DASH_REPO_ROOT: "rel/repo" });
    expect(path.isAbsolute(c.stateDir)).toBe(true);
    expect(c.stateDir).toBe(path.resolve("rel/state"));
    expect(c.repoRoot).toBe(path.resolve("rel/repo"));
    expect(c.kanbanPath).toBe(path.join(path.resolve("rel/repo"), "phase-05-kanban.md"));
    expect(loadConfig({ DASH_KANBAN: "k.md" }).kanbanPath).toBe(path.resolve("k.md"));
  });
  it("rejects a non-numeric port", () => {
    expect(() => loadConfig({ DASH_PORT: "abc" })).toThrow(/DASH_PORT/);
  });
  it("rejects port 0, which would bind a random port while printing 0", () => {
    expect(() => loadConfig({ DASH_PORT: "0" })).toThrow(/DASH_PORT/);
    expect(() => loadConfig({ DASH_PORT: "00" })).toThrow(/DASH_PORT/);
  });
  it("defaults checkpointsDir to <repoRoot>/checkpoints and resolves DASH_CHECKPOINTS to an absolute path", () => {
    const base = loadConfig({ DASH_REPO_ROOT: "/r" });
    expect(base.checkpointsDir).toBe("/r/checkpoints");
    const custom = loadConfig({ DASH_REPO_ROOT: "/r", DASH_CHECKPOINTS: "elsewhere/ck" });
    expect(path.isAbsolute(custom.checkpointsDir)).toBe(true);
    expect(custom.checkpointsDir.endsWith("elsewhere/ck")).toBe(true);
  });

  it.each(["DASH_REPO_ROOT", "DASH_STATE_DIR", "DASH_KANBAN", "DASH_CHECKPOINTS", "DASH_GH_BIN", "DASH_PYTHON"])(
    "an empty or blank %s means unset, not the current directory",
    (key) => {
      // `DASH_CHECKPOINTS=$D npm start` with $D unset passes an empty string; that must not resolve to the cwd.
      for (const blank of ["", "   "]) expect(loadConfig({ [key]: blank })).toEqual(loadConfig({}));
    },
  );

  it("a blank DASH_PORT is still an error, because silently binding the default port could hit another instance", () => {
    expect(() => loadConfig({ DASH_PORT: "" })).toThrow(/DASH_PORT must be a port number/);
    expect(() => loadConfig({ DASH_PORT: "  " })).toThrow(/DASH_PORT must be a port number/);
  });

  it("a blank setting does not disturb the others that are set", () => {
    const c = loadConfig({ DASH_REPO_ROOT: "/r", DASH_KANBAN: "", DASH_CHECKPOINTS: "/c", DASH_STATE_DIR: "  ", DASH_GH_BIN: "/bin/gh2" });
    expect(c).toMatchObject({ repoRoot: "/r", kanbanPath: "/r/phase-05-kanban.md", checkpointsDir: "/c", ghBin: "/bin/gh2" });
    expect(c.stateDir.endsWith("/dashboard/.state")).toBe(true);
  });

  it("surrounding whitespace around a real value is ignored", () => {
    expect(loadConfig({ DASH_PORT: " 4400 " }).port).toBe(4400);
    expect(loadConfig({ DASH_REPO_ROOT: "/r", DASH_KANBAN: " /x/k.md " }).kanbanPath).toBe("/x/k.md");
    expect(loadConfig({ DASH_PYTHON: " /opt/py " }).pythonBin).toBe("/opt/py");
  });
});
