import { describe, it, expect, beforeEach, afterEach } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { createChief, CHIEF_NAME, type Chief } from "../server/agents/chief";
import { AgentError } from "../server/agents/supervisor";
import { createFakeDriver, type FakeDriverOptions } from "../server/agents/fakeDriver";
import { createApprovals, type Approvals } from "../server/agents/approvals";
import { EventBus } from "../server/events";
import { loadConfig, type Config } from "../server/config";
import type { AgentDriver, SessionStart, ToolDef } from "../server/agents/driver";
import type { DashEvent } from "../shared/types";

let root: string;
let cfg: Config;
let bus: EventBus;
let approvals: Approvals;
let events: DashEvent[];
let chief: Chief;
let ran: Record<string, unknown>[];
let starts: SessionStart[];

const tools: ToolDef[] = [
  { name: "spawn_worker", description: "d", schema: {}, handler: async (a) => (ran.push(a), { text: `started ${a.name}` }) },
  { name: "list_workers", description: "d", schema: {}, handler: async () => ({ text: "none" }) },
];

function build(fake: FakeDriverOptions = { stepMs: 0 }, driver?: AgentDriver): Chief {
  const inner = driver ?? createFakeDriver(fake);
  const recording: AgentDriver = { start: (s) => (starts.push(s), inner.start(s)) };
  approvals = createApprovals({ bus, auditPath: path.join(root, "audit.jsonl") });
  return createChief({ cfg, bus, driver: recording, approvals, tools, env: { PATH: process.env.PATH ?? "" }, notifyMs: 5 });
}
async function until<T>(f: () => T | undefined | false, ms = 3000): Promise<T> {
  const end = Date.now() + ms;
  for (;;) {
    const v = f();
    if (v) return v as T;
    if (Date.now() > end) throw new Error(`timed out; chief is ${JSON.stringify(chief.summary())}`);
    await new Promise((r) => setTimeout(r, 5));
  }
}
const waitState = (s: string) => until(() => chief.summary().state === s || undefined);

beforeEach(async () => {
  root = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), "dash-chief-")));
  cfg = loadConfig({ DASH_REPO_ROOT: path.join(root, "repo"), DASH_AGENTS_DIR: path.join(root, "agents"), DASH_STATE_DIR: path.join(root, "state"), DASH_AGENT_DRIVER: "fake" });
  fs.mkdirSync(cfg.repoRoot, { recursive: true });
  bus = new EventBus();
  events = [];
  bus.subscribe((e) => events.push(e));
  ran = [];
  starts = [];
  chief = build();
  await chief.init();
});
afterEach(async () => {
  await chief.stop();
  fs.rmSync(root, { recursive: true, force: true });
});

describe("conversation", () => {
  it("starts no session until the first message", async () => {
    expect(starts).toHaveLength(0);
    expect(chief.summary()).toMatchObject({ state: "idle", sessionId: null, model: "opus", turns: 0 });
  });

  it("answers a message and returns to idle with the exchange in the journal", async () => {
    await chief.send("what is going on?");
    await until(() => chief.summary().turns === 1 || undefined);
    await waitState("idle");
    const log = (await chief.events(0, 100)).events;
    expect(log[0]).toMatchObject({ kind: "user", text: "what is going on?" });
    expect(log.map((e) => e.kind)).toEqual(expect.arrayContaining(["text", "result"]));
    expect(chief.summary().sessionId).toMatch(/^fake-/);
  });

  it("gives the role rules with the first message only, and reuses the same session afterwards", async () => {
    await chief.send("hello there");
    await waitState("idle");
    await chief.send("and another");
    await until(() => chief.summary().turns === 2 || undefined);
    expect(starts).toHaveLength(1);
    expect(starts[0].prompt).toContain("hello there");
    expect(starts[0].prompt).toMatch(/spawn_worker/);
    expect(starts[0].prompt).toMatch(/approv/i);
    const log = (await chief.events(0, 100)).events;
    expect(log.filter((e) => e.kind === "user").map((e) => (e.kind === "user" ? e.text : ""))).toEqual(["hello there", "and another"]);
  });

  it("runs read-only with the dashboard tools and hides the state directory", async () => {
    await chief.send("hi");
    const s = starts[0];
    expect(s.options.model).toBe("opus");
    expect(s.options.cwd).toBe(path.join(cfg.agentsDir, ".chief"));
    expect(s.prompt).toContain(cfg.repoRoot);
    expect(s.tools?.map((t) => t.name)).toEqual(["spawn_worker", "list_workers"]);
    expect(s.options.settings.permissions.deny).toEqual(expect.arrayContaining(["Bash", "Edit", "Write"]));
    expect(s.confinement?.worktree).toBe(path.join(cfg.agentsDir, ".chief"));
    expect(s.confinement?.maskDirs).toEqual(expect.arrayContaining([cfg.stateDir]));
    expect(s.confinement?.gitDir).toBeUndefined();
  });

  it("rejects empty and oversized messages", async () => {
    await expect(chief.send("  ")).rejects.toMatchObject({ code: "invalid" });
    await expect(chief.send("x".repeat(8001))).rejects.toMatchObject({ code: "invalid" });
    expect(starts).toHaveLength(0);
  });
});

describe("tool calls and approvals", () => {
  const spawnScript: FakeDriverOptions = { stepMs: 0, scripts: (_t, turn) => (turn === 0 ? [[{ mcp: "spawn_worker", input: { name: "w-one", task: "t" } }]] : [[{ say: "ok" }]]) };

  it("a mutating tool waits on a card, then runs when approved", async () => {
    chief = build(spawnScript);
    await chief.init();
    await chief.send("start a worker");
    await waitState("waiting-approval");
    const [card] = approvals.pending();
    expect(card).toMatchObject({ agentId: "chief", agentName: CHIEF_NAME, tool: "mcp__dashboard__spawn_worker", input: { name: "w-one", task: "t" } });
    expect(chief.summary().pendingApprovals).toBe(1);
    expect(ran).toEqual([]);
    approvals.decide(card.id, "allow");
    await waitState("idle");
    expect(ran).toEqual([{ name: "w-one", task: "t" }]);
    const kinds = (await chief.events(0, 100)).events.map((e) => (e.kind === "approval" ? `approval:${e.outcome}` : e.kind));
    expect(kinds).toEqual(expect.arrayContaining(["approval:requested", "approval:allow", "tool-use", "tool-result"]));
  });

  it("a denied tool does not run and the chief is told", async () => {
    chief = build(spawnScript);
    await chief.init();
    await chief.send("start a worker");
    await waitState("waiting-approval");
    approvals.decide(approvals.pending()[0].id, "deny", "Not now");
    await waitState("idle");
    expect(ran).toEqual([]);
    const res = (await chief.events(0, 100)).events.find((e) => e.kind === "tool-result");
    expect(res).toMatchObject({ isError: true, output: expect.stringContaining("Not now") });
  });
});

describe("stop, reset, restart and failure", () => {
  const waitScript: FakeDriverOptions = { stepMs: 0, scripts: (t) => (t.includes("start a worker") ? [[{ mcp: "spawn_worker", input: { name: "w-one", task: "t" } }]] : [[{ say: "back" }]]) };

  it("stopping cancels pending cards, and the next message resumes the same conversation", async () => {
    chief = build(waitScript);
    await chief.init();
    await chief.send("start a worker");
    await waitState("waiting-approval");
    const sessionId = chief.summary().sessionId;
    await chief.stop();
    expect(chief.summary().state).toBe("stopped");
    expect(approvals.count()).toBe(0);
    await chief.send("carry on");
    await until(() => starts.length === 2 || undefined);
    expect(starts[1].resume).toBe(sessionId);
    await waitState("idle");
  });

  it("reset forgets the conversation: a new session starts without resume", async () => {
    await chief.send("first");
    await waitState("idle");
    const first = chief.summary().sessionId;
    await chief.reset();
    expect(chief.summary()).toMatchObject({ state: "idle", sessionId: null, turns: 0 });
    await chief.send("fresh start");
    await waitState("idle");
    expect(starts[1].resume).toBeUndefined();
    expect(chief.summary().sessionId).not.toBe(first);
    expect(starts[1].prompt).toMatch(/spawn_worker/);
  });

  it("keeps its journal and session across a restart and resumes on the next message", async () => {
    await chief.send("remember me");
    await waitState("idle");
    const sessionId = chief.summary().sessionId;
    await chief.stop();
    chief = build();
    await chief.init();
    expect(chief.summary()).toMatchObject({ state: "idle", sessionId });
    expect((await chief.events(0, 100)).events[0]).toMatchObject({ kind: "user", text: "remember me" });
    await chief.send("still there?");
    await until(() => starts.length === 2 || undefined);
    expect(starts[1].resume).toBe(sessionId);
  });

  it("marks a crashed session failed with a fixed message and recovers on the next message", async () => {
    let first = true;
    const flaky: AgentDriver = { start: (s) => (first ? ((first = false), createFakeDriver({ stepMs: 0, scripts: () => [[{ fail: "secret detail /home/x" }]] }).start(s)) : createFakeDriver({ stepMs: 0 }).start(s)) };
    chief = build({}, flaky);
    await chief.init();
    await chief.send("hi");
    await waitState("failed");
    expect(chief.summary().error).toBe("The chief of staff session ended with an error.");
    expect(JSON.stringify(await chief.events(0, 100))).not.toContain("secret detail");
    await chief.send("try again");
    await waitState("idle");
    expect(chief.summary().error).toBeNull();
  });

  it("stop is safe when nothing has started", async () => {
    await expect(chief.stop()).resolves.toBeUndefined();
    expect(chief.summary().state).toBe("idle");
  });
});

describe("robustness", () => {
  it("answers safely before init has finished", async () => {
    const early = build();
    expect(early.summary()).toMatchObject({ state: "idle", lastSeq: 0 });
    expect(await early.events(0, 10)).toEqual({ events: [], lastSeq: 0 });
    await expect(early.send("hi")).rejects.toMatchObject({ code: "state" });
  });

  it("new conversation archives the log and starts a fresh one", async () => {
    await chief.send("old topic");
    await waitState("idle");
    await chief.reset();
    const fresh = (await chief.events(0, 100)).events;
    expect(fresh.some((e) => e.kind === "user")).toBe(false);
    expect(fresh[0]).toMatchObject({ kind: "state", detail: "New conversation" });
    const archived = fs.readdirSync(path.join(cfg.stateDir, "chief")).filter((n) => /^journal-.*\.jsonl$/.test(n));
    expect(archived).toHaveLength(1);
    expect(fs.readFileSync(path.join(cfg.stateDir, "chief", archived[0]), "utf8")).toContain("old topic");
  });

  it("a session that finishes after a reset cannot bring the old conversation back", async () => {
    let release!: () => void;
    const gate = new Promise<void>((r) => (release = r));
    const late: AgentDriver = {
      start: () => ({
        events: (async function* () { await gate; yield { sessionId: "old-session" }; yield { body: { kind: "result" as const, ok: true, text: "", turns: 1, costUsd: 0.5 } }; })(),
        send() {}, interrupt: async () => {}, close() {},
      }),
    };
    chief = build({}, late);
    await chief.init();
    await chief.send("hi");
    await chief.reset();
    release();
    await new Promise((r) => setTimeout(r, 50));
    expect(chief.summary()).toMatchObject({ sessionId: null, turns: 0 });
  });

  it("tools given to the session refuse to run once the chief is stopped", async () => {
    await chief.send("hi");
    await waitState("idle");
    await chief.stop();
    const tool = starts[0].tools?.find((t) => t.name === "spawn_worker") as ToolDef;
    const out = await tool.handler({ name: "late-one", task: "t" });
    expect(out).toMatchObject({ isError: true, text: expect.stringContaining("stopped") });
    expect(ran).toEqual([]);
  });
});

describe("notifications", () => {
  it("announces changes on the chat resource", async () => {
    await chief.send("hi");
    await waitState("idle");
    await new Promise((r) => setTimeout(r, 30));
    expect(events.some((e) => e.type === "invalidate" && e.resource === "chat")).toBe(true);
  });
});

describe("AgentError reuse", () => {
  it("send rejects with the shared error type", async () => {
    await expect(chief.send("")).rejects.toBeInstanceOf(AgentError);
  });
});
