import { describe, it, expect, beforeEach, afterEach } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { createWorktrees, validAgentName, type Worktrees } from "../server/agents/worktrees";

const git = (cwd: string, ...args: string[]) =>
  execFileSync("git", args, { cwd, encoding: "utf8", env: { ...process.env, GIT_CONFIG_GLOBAL: "/dev/null", GIT_CONFIG_SYSTEM: "/dev/null" } }).trim();

let root: string;
let repo: string;
let agentsDir: string;
let wt: Worktrees;

beforeEach(() => {
  root = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), "dash-wt-")));
  repo = path.join(root, "repo");
  agentsDir = path.join(root, "repo-agents");
  fs.mkdirSync(repo);
  git(repo, "init", "-q", "-b", "phase-05");
  git(repo, "config", "user.email", "t@example.com");
  git(repo, "config", "user.name", "T");
  fs.writeFileSync(path.join(repo, "a.txt"), "one\n");
  git(repo, "add", "a.txt");
  git(repo, "commit", "-q", "-m", "init");
  wt = createWorktrees({ repoRoot: repo, agentsDir, base: "phase-05" });
});
afterEach(() => fs.rmSync(root, { recursive: true, force: true }));

describe("validAgentName", () => {
  it.each(["ab", "fix-lcn", "a1", "x".repeat(31)])("accepts %s", (n) => expect(validAgentName(n)).toBe(true));
  it.each(["chief", "", "a", "-ab", "Ab", "a_b", "a/b", "..", "a b", "x".repeat(32), "ab\n"])("rejects %j", (n) => expect(validAgentName(n)).toBe(false));
});

describe("create", () => {
  it("adds a worktree on a new branch from the base, without touching the main checkout", async () => {
    const before = { head: git(repo, "rev-parse", "HEAD"), branch: git(repo, "branch", "--show-current"), status: git(repo, "status", "--porcelain") };
    const made = await wt.create("fix-lcn");
    expect(made.path).toBe(path.join(agentsDir, "fix-lcn"));
    expect(made.branch).toBe("phase-05-agent-fix-lcn");
    expect(fs.existsSync(path.join(made.path, "a.txt"))).toBe(true);
    expect(git(made.path, "branch", "--show-current")).toBe("phase-05-agent-fix-lcn");
    expect(git(repo, "rev-parse", "HEAD")).toBe(before.head);
    expect(git(repo, "branch", "--show-current")).toBe(before.branch);
    expect(git(repo, "status", "--porcelain")).toBe(before.status);
  });

  it("rejects bad names without running git", async () => {
    await expect(wt.create("../evil")).rejects.toThrow(/name/i);
    expect(fs.existsSync(agentsDir)).toBe(false);
  });

  it("refuses an existing worktree path", async () => {
    await wt.create("one-agent");
    await expect(wt.create("one-agent")).rejects.toThrow(/exists/i);
  });

  it("refuses an existing branch", async () => {
    git(repo, "branch", "phase-05-agent-taken");
    await expect(wt.create("taken")).rejects.toThrow(/exists/i);
    expect(fs.existsSync(path.join(agentsDir, "taken"))).toBe(false);
  });

  it("refuses when the agents dir is a symlink out of place", async () => {
    const elsewhere = path.join(root, "elsewhere");
    fs.mkdirSync(elsewhere);
    fs.mkdirSync(agentsDir);
    fs.symlinkSync(elsewhere, path.join(agentsDir, "linked"));
    await expect(wt.create("linked")).rejects.toThrow(/exists/i);
  });

  it("fails clearly when the base ref is missing", async () => {
    const bad = createWorktrees({ repoRoot: repo, agentsDir, base: "origin/nope" });
    await expect(bad.create("some-agent")).rejects.toThrow(/base/i);
    expect(fs.existsSync(path.join(agentsDir, "some-agent"))).toBe(false);
  });
});

describe("list and remove", () => {
  it("lists only agent worktrees", async () => {
    await wt.create("alpha");
    await wt.create("beta");
    const names = (await wt.list()).map((w) => w.name).sort();
    expect(names).toEqual(["alpha", "beta"]);
  });

  it("removes a clean worktree whose work is merged, and its branch", async () => {
    const made = await wt.create("done-one");
    await wt.remove("done-one");
    expect(fs.existsSync(made.path)).toBe(false);
    expect(git(repo, "branch", "--list", made.branch)).toBe("");
    expect(git(repo, "branch", "--show-current")).toBe("phase-05");
  });

  it("refuses to remove a worktree with uncommitted changes", async () => {
    const made = await wt.create("dirty-one");
    fs.writeFileSync(path.join(made.path, "new.txt"), "x");
    await expect(wt.remove("dirty-one")).rejects.toThrow(/uncommitted/i);
    expect(fs.existsSync(made.path)).toBe(true);
  });

  it("refuses to remove a worktree with unmerged commits unless forced", async () => {
    const made = await wt.create("ahead-one");
    fs.writeFileSync(path.join(made.path, "b.txt"), "x");
    git(made.path, "add", "b.txt");
    git(made.path, "-c", "user.email=t@example.com", "-c", "user.name=T", "commit", "-q", "-m", "work");
    await expect(wt.remove("ahead-one")).rejects.toThrow(/unmerged/i);
    await wt.remove("ahead-one", { force: true });
    expect(fs.existsSync(made.path)).toBe(false);
  });

  it("rejects removing an unknown or invalid name", async () => {
    await expect(wt.remove("nope-nope")).rejects.toThrow(/not found/i);
    await expect(wt.remove("../x")).rejects.toThrow(/name/i);
  });
});
