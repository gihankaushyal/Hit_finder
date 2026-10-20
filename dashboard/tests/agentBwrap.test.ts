import { describe, it, expect, beforeEach, afterEach } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { execFileSync, spawnSync } from "node:child_process";
import { bwrapArgs, guardedGitPaths, bwrapAvailable } from "../server/agents/bwrap";

const profile = {
  worktree: "/r/agents/a",
  gitDir: "/r/repo/.git",
  ownBranch: "phase-05-agent-a",
  home: "/home/u",
  remote: "/r/origin.git" as string | undefined,
  cwd: "/r/agents/a",
  command: "claude",
  args: ["--print"],
  guarded: ["/r/repo/.git/hooks", "/r/repo/.git/config"],
  maskDirs: ["/r/repo/.secrets"],
};

describe("bwrapArgs", () => {
  const a = bwrapArgs(profile);
  const pair = (flag: string, x: string, y = x) => a.some((v, i) => v === flag && a[i + 1] === x && a[i + 2] === y);

  it("starts from a read-only view of the whole filesystem", () => {
    expect(a.slice(0, 3)).toEqual(["--ro-bind", "/", "/"]);
  });

  it("isolates pids, dies with the parent and detaches the terminal", () => {
    expect(a).toEqual(expect.arrayContaining(["--unshare-pid", "--die-with-parent", "--new-session", "--proc", "/proc", "--dev", "/dev"]));
  });

  it("gives write access only to the worktree, the repo .git, the remote and ~/.claude", () => {
    const binds = a.flatMap((v, i) => (v === "--bind" ? [a[i + 1]] : []));
    expect(binds).toEqual(["/r/agents/a", "/r/repo/.git", "/r/origin.git", "/home/u/.claude"]);
  });

  it("uses a private /tmp", () => {
    expect(a.join(" ")).toContain("--tmpfs /tmp");
  });

  it("re-mounts guarded git paths read-only after the writable .git bind", () => {
    expect(pair("--ro-bind", "/r/repo/.git/hooks")).toBe(true);
    expect(a.indexOf("/r/repo/.git/hooks")).toBeGreaterThan(a.indexOf("/r/repo/.git"));
  });

  it("masks secret directories with an empty tmpfs", () => {
    const i = a.indexOf("/r/repo/.secrets");
    expect(a[i - 1]).toBe("--tmpfs");
  });

  it("omits the remote when there is none", () => {
    const b = bwrapArgs({ ...profile, remote: undefined });
    expect(b.flatMap((v, i) => (v === "--bind" ? [b[i + 1]] : []))).toEqual(["/r/agents/a", "/r/repo/.git", "/home/u/.claude"]);
  });

  it("ends with chdir, a separator and the command", () => {
    expect(a.slice(-5)).toEqual(["--chdir", "/r/agents/a", "--", "claude", "--print"]);
  });
});

describe("guardedGitPaths", () => {
  let root: string;
  let repo: string;
  beforeEach(() => {
    root = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), "dash-bw-")));
    repo = path.join(root, "repo");
    fs.mkdirSync(repo);
    const g = (...a: string[]) => execFileSync("git", ["-C", repo, ...a], { encoding: "utf8" });
    g("init", "-q", "-b", "phase-05");
    g("config", "user.email", "t@example.com");
    g("config", "user.name", "T");
    fs.writeFileSync(path.join(repo, "a"), "1");
    g("add", "a");
    g("commit", "-q", "-m", "i");
    g("branch", "phase-05-agent-a");
    g("branch", "other");
    g("tag", "v1");
  });
  afterEach(() => fs.rmSync(root, { recursive: true, force: true }));

  it("guards hooks, config, HEAD, info and every branch and tag ref except the agent's own", () => {
    const gd = path.join(repo, ".git");
    const guarded = guardedGitPaths(gd, "phase-05-agent-a");
    expect(guarded).toEqual(
      expect.arrayContaining([path.join(gd, "hooks"), path.join(gd, "config"), path.join(gd, "HEAD"), path.join(gd, "refs/heads/phase-05"), path.join(gd, "refs/heads/other"), path.join(gd, "refs/tags/v1")]),
    );
    expect(guarded).not.toContain(path.join(gd, "refs/heads/phase-05-agent-a"));
  });

  it("only lists paths that exist", () => {
    for (const p of guardedGitPaths(path.join(repo, ".git"), "phase-05-agent-a")) expect(fs.existsSync(p)).toBe(true);
  });
});

const haveBwrap = bwrapAvailable();
describe.skipIf(!haveBwrap)("real bwrap", () => {
  it("blocks writes outside the worktree and allows them inside", () => {
    const root = fs.realpathSync(fs.mkdtempSync(path.join(os.homedir(), ".dash-bw-test-")));
    try {
      const wt = path.join(root, "wt");
      const outside = path.join(root, "outside");
      fs.mkdirSync(wt);
      fs.mkdirSync(outside);
      const { gitDir: _g, ...rest } = profile;
      const args = bwrapArgs({ ...rest, worktree: wt, remote: undefined, cwd: wt, command: "bash", args: ["-c", `echo in > ${wt}/in.txt; echo out > ${outside}/out.txt; echo $?`], guarded: [], maskDirs: [], home: os.homedir() });
      const r = spawnSync("bwrap", args, { encoding: "utf8" });
      expect(fs.existsSync(path.join(wt, "in.txt"))).toBe(true);
      expect(fs.existsSync(path.join(outside, "out.txt"))).toBe(false);
      expect(r.status).toBe(0);
    } finally {
      fs.rmSync(root, { recursive: true, force: true });
    }
  });
});
