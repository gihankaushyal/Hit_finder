import { describe, it, expect, beforeEach, afterEach } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { createSupervisor, AgentError, type Supervisor } from "../server/agents/supervisor";
import { createFakeDriver } from "../server/agents/fakeDriver";
import { createApprovals, type Approvals } from "../server/agents/approvals";
import { createWorktrees } from "../server/agents/worktrees";
import { EventBus } from "../server/events";
import { loadConfig, type Config } from "../server/config";
import type { AgentDriver, SessionStart } from "../server/agents/driver";
import type { AgentSummary, DashEvent } from "../shared/types";

const git = (cwd: string, ...a: string[]) => execFileSync("git", a, { cwd, encoding: "utf8" }).trim();

let root: string;
let repo: string;
let cfg: Config;
let bus: EventBus;
let approvals: Approvals;
let events: DashEvent[];
let sup: Supervisor;

function build(env: Record<string, string> = {}, driver: AgentDriver = createFakeDriver({ stepMs: 0 })): Supervisor {
  cfg = loadConfig({ DASH_REPO_ROOT: repo, DASH_AGENTS_DIR: path.join(root, "agents"), DASH_STATE_DIR: path.join(root, "state"), DASH_AGENT_BASE: "phase-05", ...env });
  approvals = createApprovals({ bus, auditPath: path.join(root, "audit.jsonl") });
  return createSupervisor({ cfg, bus, driver, approvals, worktrees: createWorktrees({ repoRoot: repo, agentsDir: cfg.agentsDir, base: cfg.agentBase }), notifyMs: 5, env: { PATH: process.env.PATH ?? "" } });
}

async function until<T>(f: () => T | undefined | false, ms = 3000): Promise<T> {
  const end = Date.now() + ms;
  for (;;) {
    const v = f();
    if (v) return v as T;
    if (Date.now() > end) throw new Error("timed out waiting");
    await new Promise((r) => setTimeout(r, 5));
  }
}
const stateOf = (id: string) => sup.get(id)?.state;
const waitState = async (id: string, s: string) => {
  try {
    return await until(() => stateOf(id) === s || undefined);
  } catch (e) {
    throw new Error(`waiting for ${s}; agent is ${JSON.stringify(sup.get(id))}; journal ${JSON.stringify((await sup.events(id, 0, 50)).events)}`, { cause: e });
  }
};

beforeEach(async () => {
  root = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), "dash-sup-")));
  repo = path.join(root, "repo");
  fs.mkdirSync(repo);
  git(repo, "init", "-q", "-b", "phase-05");
  git(repo, "config", "user.email", "t@example.com");
  git(repo, "config", "user.name", "T");
  fs.writeFileSync(path.join(repo, "a.txt"), "1");
  git(repo, "add", "a.txt");
  git(repo, "commit", "-q", "-m", "init");
  bus = new EventBus();
  events = [];
  bus.subscribe((e) => events.push(e));
  sup = build();
  await sup.init();
});
afterEach(async () => {
  await sup.stopAll(1000);
  fs.rmSync(root, { recursive: true, force: true });
});

describe("spawn", () => {
  it("creates a worktree, runs the task and ends idle", async () => {
    const a = await sup.spawn({ name: "fix-lcn", task: "tidy the notes" });
    expect(a).toMatchObject({ name: "fix-lcn", role: "worker", branch: "phase-05-agent-fix-lcn", worktree: path.join(cfg.agentsDir, "fix-lcn"), model: "sonnet" });
    await waitState(a.id, "idle");
    const s = sup.get(a.id) as AgentSummary;
    expect(s.turns).toBe(1);
    expect(s.sessionId).toMatch(/^fake-/);
    expect(fs.existsSync(path.join(s.worktree as string, "a.txt"))).toBe(true);
    const log = await sup.events(a.id, 0, 100);
    expect(log.events.map((e) => e.kind)).toEqual(expect.arrayContaining(["user", "state", "text", "result"]));
    expect(log.events[0]).toMatchObject({ kind: "user", text: "tidy the notes" });
  });

  it("gives the agent the task and its branch in the prompt", async () => {
    let prompt = "";
    const capture: AgentDriver = { start: (s: SessionStart) => ((prompt = s.prompt), createFakeDriver({ stepMs: 0 }).start(s)) };
    await sup.stopAll(500);
    sup = build({}, capture);
    await sup.init();
    await sup.spawn({ name: "cap-agent", task: "do exactly this" });
    expect(prompt).toContain("do exactly this");
    expect(prompt).toContain("phase-05-agent-cap-agent");
  });

  it("links an issue number", async () => {
    const a = await sup.spawn({ name: "linked", task: "t", issue: 77 });
    expect(a.issue).toBe(77);
  });

  it("rejects bad input with an invalid error", async () => {
    await expect(sup.spawn({ name: "Bad Name", task: "t" })).rejects.toMatchObject({ code: "invalid" });
    await expect(sup.spawn({ name: "good-name", task: "   " })).rejects.toMatchObject({ code: "invalid" });
    await expect(sup.spawn({ name: "good-name", task: "x".repeat(8001) })).rejects.toMatchObject({ code: "invalid" });
    await expect(sup.spawn({ name: "good-name", task: "t", issue: -1 })).rejects.toMatchObject({ code: "invalid" });
  });

  it("refuses a second agent with the same name", async () => {
    await sup.spawn({ name: "same-name", task: "t" });
    await expect(sup.spawn({ name: "same-name", task: "t" })).rejects.toMatchObject({ code: "exists" });
  });

  it("enforces the concurrency cap and frees a slot when an agent stops", async () => {
    await sup.stopAll(500);
    sup = build({ DASH_AGENT_MAX: "1" });
    await sup.init();
    const a = await sup.spawn({ name: "first-one", task: "t" });
    await expect(sup.spawn({ name: "second-one", task: "t" })).rejects.toMatchObject({ code: "cap" });
    await sup.stop(a.id);
    await expect(sup.spawn({ name: "second-one", task: "t" })).resolves.toMatchObject({ name: "second-one" });
  });

  it("does not touch the main checkout", async () => {
    const before = [git(repo, "rev-parse", "HEAD"), git(repo, "branch", "--show-current"), git(repo, "status", "--porcelain")];
    const a = await sup.spawn({ name: "quiet-one", task: "t" });
    await waitState(a.id, "idle");
    await sup.stop(a.id);
    expect([git(repo, "rev-parse", "HEAD"), git(repo, "branch", "--show-current"), git(repo, "status", "--porcelain")]).toEqual(before);
  });
});

describe("approvals", () => {
  it("waits for a decision on a gated tool, then continues", async () => {
    const a = await sup.spawn({ name: "pusher", task: "push the branch" });
    await waitState(a.id, "waiting-approval");
    const [card] = approvals.pending();
    expect(card).toMatchObject({ agentId: a.id, agentName: "pusher", tool: "Bash", input: { command: "git push origin HEAD" } });
    expect(sup.get(a.id)?.pendingApprovals).toBe(1);
    approvals.decide(card.id, "allow");
    await waitState(a.id, "idle");
    const kinds = (await sup.events(a.id, 0, 100)).events.map((e) => (e.kind === "approval" ? `approval:${e.outcome}` : e.kind));
    expect(kinds).toEqual(expect.arrayContaining(["approval:requested", "approval:allow", "tool-result"]));
  });

  it("tells the agent when a request is denied", async () => {
    const a = await sup.spawn({ name: "denied-one", task: "push the branch" });
    await waitState(a.id, "waiting-approval");
    approvals.decide(approvals.pending()[0].id, "deny", "Not yet");
    await waitState(a.id, "idle");
    const log = (await sup.events(a.id, 0, 100)).events;
    expect(log.find((e) => e.kind === "tool-result")).toMatchObject({ isError: true, output: expect.stringContaining("Not yet") });
  });

  it("stopping an agent cancels its pending approvals", async () => {
    const a = await sup.spawn({ name: "stopped-one", task: "push the branch" });
    await waitState(a.id, "waiting-approval");
    const id = approvals.pending()[0].id;
    await sup.stop(a.id);
    expect(stateOf(a.id)).toBe("stopped");
    expect(approvals.count()).toBe(0);
    expect(() => approvals.decide(id, "allow")).toThrow();
  });
});

describe("message and stop", () => {
  it("runs another turn for a message and returns to idle", async () => {
    const a = await sup.spawn({ name: "chatty", task: "t" });
    await waitState(a.id, "idle");
    await sup.message(a.id, "second thing");
    await until(() => sup.get(a.id)?.turns === 2 || undefined);
    await waitState(a.id, "idle");
    const log = (await sup.events(a.id, 0, 100)).events;
    expect(log.filter((e) => e.kind === "user").map((e) => (e.kind === "user" ? e.text : ""))).toEqual(["t", "second thing"]);
  });

  it("rejects an empty or oversized message and an unknown agent", async () => {
    const a = await sup.spawn({ name: "picky-one", task: "t" });
    await expect(sup.message(a.id, "  ")).rejects.toMatchObject({ code: "invalid" });
    await expect(sup.message(a.id, "x".repeat(8001))).rejects.toMatchObject({ code: "invalid" });
    await expect(sup.message("nope-nope", "hi")).rejects.toMatchObject({ code: "not-found" });
  });

  it("refuses to message a stopped agent", async () => {
    const a = await sup.spawn({ name: "gone-one", task: "t" });
    await sup.stop(a.id);
    await expect(sup.message(a.id, "hi")).rejects.toMatchObject({ code: "state" });
  });

  it("stop is idempotent", async () => {
    const a = await sup.spawn({ name: "twice-one", task: "t" });
    await sup.stop(a.id);
    await expect(sup.stop(a.id)).resolves.toBeUndefined();
    expect(stateOf(a.id)).toBe("stopped");
  });
});

describe("failure", () => {
  it("marks a crashed session failed with a fixed message and keeps the worktree", async () => {
    const a = await sup.spawn({ name: "crashy", task: "this will fail" });
    await waitState(a.id, "failed");
    const s = sup.get(a.id) as AgentSummary;
    expect(s.error).toBe("The agent session ended with an error.");
    expect(JSON.stringify(s)).not.toContain("Scripted failure");
    expect(fs.existsSync(s.worktree as string)).toBe(true);
    const log = (await sup.events(a.id, 0, 100)).events;
    expect(log.find((e) => e.kind === "error")).toMatchObject({ message: "The agent session ended with an error." });
  });
});

describe("events and notifications", () => {
  it("reads the journal by cursor", async () => {
    const a = await sup.spawn({ name: "cursor-one", task: "t" });
    await waitState(a.id, "idle");
    const all = await sup.events(a.id, 0, 100);
    const rest = await sup.events(a.id, all.events[1].seq, 100);
    expect(rest.events[0].seq).toBe(all.events[2].seq);
    expect(all.lastSeq).toBe(all.events[all.events.length - 1].seq);
    await expect(sup.events("nope-nope", 0, 10)).rejects.toMatchObject({ code: "not-found" });
  });

  it("emits coalesced agents invalidations", async () => {
    const a = await sup.spawn({ name: "notify-one", task: "t" });
    await waitState(a.id, "idle");
    await new Promise((r) => setTimeout(r, 30));
    const n = events.filter((e) => e.type === "invalidate" && e.resource === "agents").length;
    expect(n).toBeGreaterThan(0);
    expect(n).toBeLessThan(10);
  });
});

describe("shutdown and restart", () => {
  it("stopAll stops every running agent", async () => {
    const a = await sup.spawn({ name: "one-one", task: "push it" });
    const b = await sup.spawn({ name: "two-two", task: "push it" });
    await waitState(a.id, "waiting-approval");
    await waitState(b.id, "waiting-approval");
    await sup.stopAll(1000);
    expect([stateOf(a.id), stateOf(b.id)]).toEqual(["stopped", "stopped"]);
    expect(approvals.count()).toBe(0);
  });

  it("reloads agents after a restart as stopped and can resume them", async () => {
    const a = await sup.spawn({ name: "keeper", task: "t" });
    await waitState(a.id, "idle");
    const sessionId = (sup.get(a.id) as AgentSummary).sessionId;
    await sup.stopAll(500);
    sup = build();
    await sup.init();
    expect(sup.get(a.id)).toMatchObject({ state: "stopped", sessionId });
    expect((await sup.events(a.id, 0, 100)).events.length).toBeGreaterThan(0);
    await sup.resume(a.id);
    await waitState(a.id, "idle");
    expect((sup.get(a.id) as AgentSummary).sessionId).toBe(sessionId);
  });

  it("refuses to resume an agent that is running", async () => {
    const a = await sup.spawn({ name: "busy-one", task: "push it" });
    await waitState(a.id, "waiting-approval");
    await expect(sup.resume(a.id)).rejects.toMatchObject({ code: "state" });
  });
});

describe("AgentError", () => {
  it("carries a code", () => {
    expect(new AgentError("cap", "full").code).toBe("cap");
  });
});
