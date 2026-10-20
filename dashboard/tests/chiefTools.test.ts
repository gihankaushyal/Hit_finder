import { describe, it, expect, vi } from "vitest";
import { createChiefTools } from "../server/agents/chiefTools";
import { AgentError } from "../server/agents/supervisor";
import { NotImportedError, TaskNotFoundError } from "../server/lib/kanbanService";
import type { AgentSummary } from "../shared/types";

const summary = (name: string, over: Partial<AgentSummary> = {}): AgentSummary => ({
  id: name, name, role: "worker", task: `task of ${name}`, issue: null, state: "idle", model: "sonnet", branch: `phase-05-agent-${name}`, worktree: `/w/${name}`,
  createdAt: "", updatedAt: "", lastActivity: null, turns: 2, costUsd: null, sessionId: "s", error: null, pendingApprovals: 0, lastSeq: 1, ...over,
});

function rig() {
  const supervisor = {
    list: vi.fn(() => [summary("one-one"), summary("two-two", { state: "waiting-approval", pendingApprovals: 1 })]),
    spawn: vi.fn(async (i: { name: string }) => summary(i.name)),
    message: vi.fn(async () => {}),
    stop: vi.fn(async () => {}),
    get: vi.fn((id: string) => (id === "one-one" ? summary("one-one") : undefined)),
  };
  const kanban = { addTask: vi.fn(async () => {}), setStatus: vi.fn(async () => {}) };
  const tools = createChiefTools({ supervisor: supervisor as never, kanban: kanban as never });
  const call = (name: string, args: Record<string, unknown>) => (tools.find((t) => t.name === name) as NonNullable<(typeof tools)[number]>).handler(args);
  return { supervisor, kanban, tools, call };
}

describe("chief tools", () => {
  it("offers exactly the six tools", () => {
    expect(rig().tools.map((t) => t.name).sort()).toEqual(["create_task", "list_workers", "message_worker", "spawn_worker", "stop_worker", "update_task"]);
  });

  it("list_workers describes each worker, one per line", async () => {
    const out = await rig().call("list_workers", {});
    expect(out.isError).toBeFalsy();
    const lines = out.text.split("\n");
    expect(lines).toHaveLength(2);
    expect(lines[0]).toContain("one-one");
    expect(lines[1]).toContain("waiting-approval");
    expect(lines[1]).toContain("1 pending");
  });

  it("list_workers says so when there are none", async () => {
    const r = rig();
    r.supervisor.list.mockReturnValue([]);
    expect((await r.call("list_workers", {})).text).toMatch(/no workers/i);
  });

  it("spawn_worker starts a worker and reports its branch", async () => {
    const r = rig();
    const out = await r.call("spawn_worker", { name: "new-one", task: "do it", issue: 5 });
    expect(r.supervisor.spawn).toHaveBeenCalledWith({ name: "new-one", task: "do it", issue: 5 });
    expect(out.text).toContain("phase-05-agent-new-one");
  });

  it("reports a refused spawn (cap, name) as an error the chief can read", async () => {
    const r = rig();
    r.supervisor.spawn.mockRejectedValue(new AgentError("cap", "3 agents are already running. Stop one first."));
    const out = await r.call("spawn_worker", { name: "new-one", task: "t" });
    expect(out).toEqual({ text: "3 agents are already running. Stop one first.", isError: true });
  });

  it("rejects invalid arguments without calling the supervisor", async () => {
    const r = rig();
    for (const args of [{ name: "Bad Name", task: "t" }, { name: "ok-name" }, { name: "ok-name", task: "t", issue: -2 }, { name: 5, task: "t" }]) {
      const out = await r.call("spawn_worker", args);
      expect(out.isError).toBe(true);
    }
    expect(r.supervisor.spawn).not.toHaveBeenCalled();
  });

  it("message_worker and stop_worker go through the supervisor", async () => {
    const r = rig();
    expect((await r.call("message_worker", { name: "one-one", text: "hello" })).isError).toBeFalsy();
    expect(r.supervisor.message).toHaveBeenCalledWith("one-one", "hello");
    expect((await r.call("stop_worker", { name: "one-one" })).isError).toBeFalsy();
    expect(r.supervisor.stop).toHaveBeenCalledWith("one-one");
  });

  it("stop_worker on an unknown worker is a readable error, not a crash", async () => {
    const out = await rig().call("stop_worker", { name: "ghost-one" });
    expect(out).toMatchObject({ isError: true, text: expect.stringContaining("ghost-one") });
  });

  it("an unexpected failure shows a fixed message, never the raw error", async () => {
    const r = rig();
    r.supervisor.message.mockRejectedValue(new Error("EACCES /home/someone/secret"));
    const out = await r.call("message_worker", { name: "one-one", text: "x" });
    expect(out).toEqual({ text: "The request failed. See the dashboard log.", isError: true });
  });

  it("create_task adds a kanban item with the title and body", async () => {
    const r = rig();
    const out = await r.call("create_task", { title: "Fix LCN", body: "details" });
    expect(r.kanban.addTask).toHaveBeenCalledWith("Fix LCN", "details");
    expect(out.isError).toBeFalsy();
  });

  it("create_task validates the title and passes on the not-imported message", async () => {
    const r = rig();
    expect((await r.call("create_task", { title: "" })).isError).toBe(true);
    expect((await r.call("create_task", { title: "x".repeat(201) })).isError).toBe(true);
    expect(r.kanban.addTask).not.toHaveBeenCalled();
    r.kanban.addTask.mockRejectedValue(new NotImportedError());
    const out = await r.call("create_task", { title: "ok" });
    expect(out).toMatchObject({ isError: true, text: expect.stringContaining("not been imported") });
  });

  it("update_task sets a status, rejects unknown statuses, reports a missing task", async () => {
    const r = rig();
    expect((await r.call("update_task", { number: 12, status: "done" })).isError).toBeFalsy();
    expect(r.kanban.setStatus).toHaveBeenCalledWith(12, "done");
    expect((await r.call("update_task", { number: 12, status: "finished" })).isError).toBe(true);
    expect((await r.call("update_task", { number: 0, status: "done" })).isError).toBe(true);
    r.kanban.setStatus.mockRejectedValue(new TaskNotFoundError(99));
    expect(await r.call("update_task", { number: 99, status: "done" })).toMatchObject({ isError: true, text: "task #99 not found" });
  });
});
