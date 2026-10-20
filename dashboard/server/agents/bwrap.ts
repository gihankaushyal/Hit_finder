import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";

/**
 * Confinement for one agent session, built from bubblewrap (the SDK's own sandbox stalls on this node).
 * The whole filesystem is visible read-only; only the worktree, the repository's `.git`, a local
 * remote and `~/.claude` are writable, `/tmp` is private and `.secrets` directories are hidden.
 * The main checkout is therefore read-only to every agent.
 */
export interface BwrapProfile {
  worktree: string;
  /** The main repository's `.git`: needed to commit, so writable, with sensitive parts re-mounted read-only. */
  gitDir?: string;
  ownBranch?: string;
  home: string;
  /** A bare remote on local disk that an approved push may write to. */
  remote?: string;
  cwd: string;
  command: string;
  args: string[];
  /** Paths re-mounted read-only on top of the writable `.git`. */
  guarded: string[];
  /** Directories replaced with an empty tmpfs. */
  maskDirs: string[];
}

export function bwrapArgs(p: BwrapProfile): string[] {
  const a = ["--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--unshare-pid", "--die-with-parent", "--new-session", "--tmpfs", "/tmp"];
  a.push("--bind", p.worktree, p.worktree);
  if (p.gitDir) a.push("--bind", p.gitDir, p.gitDir);
  if (p.remote) a.push("--bind", p.remote, p.remote);
  const claudeDir = path.join(p.home, ".claude");
  a.push("--bind", claudeDir, claudeDir);
  for (const g of p.guarded) a.push("--ro-bind", g, g);
  for (const m of p.maskDirs) a.push("--tmpfs", m);
  a.push("--chdir", p.cwd, "--", p.command, ...p.args);
  return a;
}

function listFiles(dir: string): string[] {
  const out: string[] = [];
  let entries: fs.Dirent[];
  try {
    entries = fs.readdirSync(dir, { withFileTypes: true });
  } catch {
    return out;
  }
  for (const e of entries) {
    const full = path.join(dir, e.name);
    if (e.isDirectory()) out.push(...listFiles(full));
    else if (e.isFile()) out.push(full);
  }
  return out;
}

/**
 * Git paths a worker must not rewrite: hooks, config, HEAD and packed-refs (command execution and
 * ref redirection), plus every branch and tag ref except its own branch. Only paths that exist are returned.
 */
export function guardedGitPaths(gitDir: string, ownBranch: string): string[] {
  const fixed = ["hooks", "config", "HEAD", "info", "packed-refs"].map((n) => path.join(gitDir, n)).filter((p) => fs.existsSync(p));
  const own = path.join(gitDir, "refs", "heads", ownBranch);
  const refs = ["heads", "tags"].flatMap((k) => listFiles(path.join(gitDir, "refs", k))).filter((p) => p !== own);
  return [...fixed, ...refs];
}

export function bwrapAvailable(): boolean {
  try {
    return spawnSync("bwrap", ["--ro-bind", "/", "/", "true"], { timeout: 5000 }).status === 0;
  } catch {
    return false;
  }
}
