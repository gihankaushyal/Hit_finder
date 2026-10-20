import fs from "node:fs/promises";
import path from "node:path";
import { execFile } from "node:child_process";
import { promisify } from "node:util";

const run = promisify(execFile);
const GIT_TIMEOUT_MS = 30_000;
const NAME_RE = /^[a-z0-9][a-z0-9-]{1,30}$/;
export const BRANCH_PREFIX = "phase-05-agent-";

export interface AgentWorktree {
  name: string;
  path: string;
  branch: string;
}

export interface Worktrees {
  create(name: string): Promise<AgentWorktree>;
  list(): Promise<AgentWorktree[]>;
  /** Removes a clean, fully merged worktree and its branch; `force` skips both checks. */
  remove(name: string, opts?: { force?: boolean }): Promise<void>;
}

/** A refusal with a message that is safe to show: it names no paths and carries no git output. */
export class WorktreeError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "WorktreeError";
  }
}

export function validAgentName(name: string): boolean {
  return NAME_RE.test(name);
}

function need(name: string): void {
  if (!validAgentName(name)) throw new WorktreeError("Agent name must be 2 to 31 characters: lowercase letters, digits and hyphens, starting with a letter or digit.");
}

async function git(cwd: string, args: string[]): Promise<string> {
  const { stdout } = await run("git", args, { cwd, timeout: GIT_TIMEOUT_MS, env: { ...process.env, GIT_TERMINAL_PROMPT: "0" }, maxBuffer: 4 * 1024 * 1024 });
  return stdout.trim();
}

async function gitOk(cwd: string, args: string[]): Promise<boolean> {
  try {
    await git(cwd, args);
    return true;
  } catch {
    return false;
  }
}

async function exists(p: string): Promise<boolean> {
  try {
    await fs.lstat(p);
    return true;
  } catch {
    return false;
  }
}

/**
 * Agent worktrees live under `agentsDir`, one per agent, on branches `phase-05-agent-<name>`.
 * Only `git worktree` and `git branch` run against the main repository, always with `-C`;
 * its working tree, HEAD and index are never touched.
 */
export function createWorktrees(cfg: { repoRoot: string; agentsDir: string; base: string }): Worktrees {
  const { repoRoot, agentsDir, base } = cfg;
  const pathFor = (name: string) => path.join(agentsDir, name);
  const branchFor = (name: string) => BRANCH_PREFIX + name;

  async function list(): Promise<AgentWorktree[]> {
    const out = await git(repoRoot, ["worktree", "list", "--porcelain"]);
    const found: AgentWorktree[] = [];
    const root = (await exists(agentsDir)) ? await fs.realpath(agentsDir) : path.resolve(agentsDir);
    for (const block of out.split("\n\n")) {
      const wtPath = /^worktree (.+)$/m.exec(block)?.[1];
      const branch = /^branch refs\/heads\/(.+)$/m.exec(block)?.[1];
      if (!wtPath || !branch?.startsWith(BRANCH_PREFIX)) continue;
      if (path.dirname(wtPath) !== root && path.dirname(wtPath) !== path.resolve(agentsDir)) continue;
      found.push({ name: path.basename(wtPath), path: pathFor(path.basename(wtPath)), branch });
    }
    return found;
  }

  return {
    list,
    async create(name) {
      need(name);
      const target = pathFor(name);
      const branch = branchFor(name);
      if (await exists(target)) throw new WorktreeError(`A worktree or file already exists for "${name}".`);
      if (await gitOk(repoRoot, ["show-ref", "--verify", "--quiet", `refs/heads/${branch}`])) throw new WorktreeError(`Branch ${branch} already exists.`);
      if (!(await gitOk(repoRoot, ["rev-parse", "--verify", "--quiet", `${base}^{commit}`]))) throw new WorktreeError(`Base ref ${base} was not found.`);
      await fs.mkdir(agentsDir, { recursive: true });
      const realAgents = await fs.realpath(agentsDir);
      if (path.dirname(path.join(realAgents, name)) !== realAgents) throw new WorktreeError("Worktree path escapes the agents directory.");
      await git(repoRoot, ["worktree", "add", "--no-track", "-b", branch, path.join(realAgents, name), base]);
      return { name, path: target, branch };
    },
    async remove(name, opts = {}) {
      need(name);
      const entry = (await list()).find((w) => w.name === name);
      if (!entry) throw new WorktreeError(`Worktree "${name}" was not found.`);
      if (!opts.force) {
        if ((await git(entry.path, ["status", "--porcelain"])) !== "") throw new WorktreeError(`Worktree "${name}" has uncommitted changes.`);
        if (Number(await git(repoRoot, ["rev-list", "--count", `${base}..${entry.branch}`])) > 0) throw new WorktreeError(`Branch ${entry.branch} has unmerged commits.`);
      }
      await git(repoRoot, ["worktree", "remove", ...(opts.force ? ["--force"] : []), entry.path]);
      await git(repoRoot, ["branch", "-D", entry.branch]);
    },
  };
}

/** The path of `origin` when it is a bare repository on local disk (an approved push may write there), else undefined. */
export async function localRemotePath(repoRoot: string): Promise<string | undefined> {
  try {
    const url = await git(repoRoot, ["config", "--get", "remote.origin.url"]);
    return path.isAbsolute(url) && (await exists(url)) ? await fs.realpath(url) : undefined;
  } catch {
    return undefined;
  }
}
