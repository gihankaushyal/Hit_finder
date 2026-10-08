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
});
