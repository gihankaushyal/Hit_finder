import fs from "node:fs/promises";
import path from "node:path";
import { execFile } from "node:child_process";
import { promisify } from "node:util";

const run = promisify(execFile);
const OURS = new Set(["claude", "socat", "bwrap"]);

export interface ProcInfo {
  pid: number;
  ppid: number;
  comm: string;
  cwd: string | null;
}

async function listProcs(): Promise<ProcInfo[]> {
  const { stdout } = await run("ps", ["-u", String(process.getuid?.() ?? ""), "-o", "pid=,ppid=,comm="], { timeout: 10_000 });
  const out: ProcInfo[] = [];
  for (const line of stdout.split("\n")) {
    const m = /^\s*(\d+)\s+(\d+)\s+(\S+)/.exec(line);
    if (!m) continue;
    const pid = Number(m[1]);
    let cwd: string | null = null;
    try {
      cwd = await fs.readlink(`/proc/${pid}/cwd`);
    } catch {
      /* gone or not ours */
    }
    out.push({ pid, ppid: Number(m[2]), comm: m[3], cwd });
  }
  return out;
}

/**
 * Kills claude, socat and bwrap processes that lost their parent (a dashboard crash) and are working
 * inside the agents directory. Processes anywhere else, or with a live parent, are never touched.
 */
export async function sweepOrphans(agentsDir: string, deps: { list?: () => Promise<ProcInfo[]>; kill?: (pid: number) => void } = {}): Promise<number[]> {
  const list = deps.list ?? listProcs;
  const kill = deps.kill ?? ((pid: number) => process.kill(pid, "SIGKILL"));
  const prefix = path.resolve(agentsDir) + path.sep;
  const found: number[] = [];
  for (const p of await list()) {
    if (p.ppid !== 1 || !OURS.has(p.comm) || !p.cwd || !(p.cwd + path.sep).startsWith(prefix)) continue;
    found.push(p.pid);
    try {
      kill(p.pid);
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code !== "ESRCH") console.error("orphan sweep kill failed", (err as NodeJS.ErrnoException).code ?? "error");
    }
  }
  return found;
}
