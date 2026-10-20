import { describe, it, expect } from "vitest";
import { sweepOrphans, type ProcInfo } from "../server/agents/orphans";

const agents = "/r/agents";
const procs: ProcInfo[] = [
  { pid: 10, ppid: 1, comm: "claude", cwd: "/r/agents/a" },
  { pid: 11, ppid: 1, comm: "socat", cwd: "/r/agents/b/sub" },
  { pid: 12, ppid: 1, comm: "bwrap", cwd: "/r/agents/a" },
  { pid: 13, ppid: 99, comm: "claude", cwd: "/r/agents/a" }, // has a live parent
  { pid: 14, ppid: 1, comm: "claude", cwd: "/r/repo" }, // someone's own session elsewhere
  { pid: 15, ppid: 1, comm: "claude", cwd: "/r/agents-other/a" }, // sibling directory with the same prefix
  { pid: 16, ppid: 1, comm: "bash", cwd: "/r/agents/a" }, // not one of ours
  { pid: 17, ppid: 1, comm: "claude", cwd: null }, // cwd unreadable
];

describe("sweepOrphans", () => {
  it("kills only parentless claude, socat and bwrap processes working inside the agents directory", async () => {
    const killed: number[] = [];
    const found = await sweepOrphans(agents, { list: async () => procs, kill: (pid) => void killed.push(pid) });
    expect(killed.sort()).toEqual([10, 11, 12]);
    expect(found.sort()).toEqual([10, 11, 12]);
  });

  it("does nothing when there are none", async () => {
    const killed: number[] = [];
    expect(await sweepOrphans(agents, { list: async () => [], kill: (p) => void killed.push(p) })).toEqual([]);
    expect(killed).toEqual([]);
  });

  it("ignores a process that is already gone", async () => {
    const found = await sweepOrphans(agents, {
      list: async () => procs.slice(0, 1),
      kill: () => {
        throw Object.assign(new Error("gone"), { code: "ESRCH" });
      },
    });
    expect(found).toEqual([10]);
  });
});
