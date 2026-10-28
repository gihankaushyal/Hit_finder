import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { createChief, CHIEF_ID, CHIEF_NAME, type Chief, type ChiefDeps } from "../server/agents/chief";
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

function build(fake: FakeDriverOptions = { stepMs: 0 }, driver?: AgentDriver, extra: Partial<ChiefDeps> = {}): Chief {
  const inner = driver ?? createFakeDriver(fake);
  const recording: AgentDriver = { start: (s) => (starts.push(s), inner.start(s)) };
  approvals = createApprovals({ bus, auditPath: path.join(root, "audit.jsonl") });
  // The fake driver's session ids are not on disk: tests get a probe that finds every transcript unless they ask otherwise.
  return createChief({ cfg, bus, driver: recording, approvals, tools, env: { PATH: process.env.PATH ?? "" }, notifyMs: 5, transcriptExists: async () => true, ...extra });
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

  it("emits agent-state for the chief, once per real change", async () => {
    await chief.send("hello");
    await until(() => chief.summary().turns === 1 || undefined);
    await waitState("idle");
    const mine = events.filter((e) => e.type === "agent-state" && e.id === CHIEF_ID) as Extract<DashEvent, { type: "agent-state" }>[];
    expect(mine.length).toBeGreaterThan(0);
    expect(mine[0].name).toBe(CHIEF_NAME);
    expect(mine.map((e) => e.state)).toContain("running");
    expect(mine[mine.length - 1].state).toBe("idle");
    expect(mine.filter((e, i) => i > 0 && e.state === mine[i - 1].state)).toEqual([]);
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

  it("a card left over from a conversation that was switched away writes nothing into the new one", async () => {
    chief = build(spawnScript);
    await chief.init();
    await chief.send("start a worker");
    await waitState("waiting-approval");
    await chief.reset();
    await new Promise((r) => setTimeout(r, 60)); // time for the cancelled card's continuation to run
    const kinds = (await chief.events(0, 100)).events.map((e) => e.kind);
    expect(kinds).not.toContain("approval");
    expect(chief.summary().state).toBe("idle");
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
    expect(await early.events(0, 10)).toEqual({ events: [], lastSeq: 0, conversationId: "" });
    await expect(early.send("hi")).rejects.toMatchObject({ code: "state" });
  });

  it("new conversation parks the old log and starts a fresh one", async () => {
    await chief.send("old topic");
    await waitState("idle");
    const old = chief.summary().conversationId;
    await chief.reset();
    const fresh = (await chief.events(0, 100)).events;
    expect(fresh.some((e) => e.kind === "user")).toBe(false);
    expect(fresh[0]).toMatchObject({ kind: "state", detail: "New conversation" });
    expect(fs.readFileSync(path.join(cfg.stateDir, "chief", "conv", old, "journal.jsonl"), "utf8")).toContain("old topic");
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

describe("conversations", () => {
  it("names a conversation after its first message", async () => {
    await chief.send("  plan   the\nweek  ");
    await waitState("idle");
    expect(chief.summary()).toMatchObject({ title: "plan the week", readOnly: null });
    expect(chief.summary().conversationId).toMatch(/^c-/);
  });

  it("keeps a parked conversation on disk with its own journal and session id", async () => {
    await chief.send("topic one");
    await waitState("idle");
    const first = chief.summary();
    expect(first.sessionId).toBeTruthy();
    await chief.reset();
    const second = chief.summary();
    expect(second).toMatchObject({ turns: 0, sessionId: null, title: null });
    expect(second.conversationId).not.toBe(first.conversationId);
    const index = JSON.parse(fs.readFileSync(path.join(cfg.stateDir, "chief", "index.json"), "utf8"));
    expect(index.active).toBe(second.conversationId);
    expect(index.convs[first.conversationId]).toMatchObject({ title: "topic one", turns: 1, sessionId: first.sessionId });
    expect(fs.readFileSync(path.join(cfg.stateDir, "chief", "conv", first.conversationId, "journal.jsonl"), "utf8")).toContain("topic one");
  });

  it("does not keep an empty conversation around", async () => {
    const before = chief.summary().conversationId;
    await chief.reset();
    await chief.reset();
    expect(chief.summary().conversationId).not.toBe(before);
    expect(fs.readdirSync(path.join(cfg.stateDir, "chief", "conv"))).toHaveLength(1);
  });

  it("comes back with the same active conversation after a restart", async () => {
    await chief.send("before the restart");
    await waitState("idle");
    const saved = chief.summary();
    const again = build();
    await again.init();
    expect(again.summary()).toMatchObject({ conversationId: saved.conversationId, title: "before the restart", turns: 1, sessionId: saved.sessionId, state: "idle" });
    expect((await again.events(0, 100)).events.some((e) => e.kind === "user" && e.text === "before the restart")).toBe(true);
  });

  it("keeps the conversation if the active pointer is lost", async () => {
    await chief.send("keep me");
    await waitState("idle");
    await chief.stop(); // the save queued by the last state change must land before the files are edited
    const saved = chief.summary();
    const file = path.join(cfg.stateDir, "chief", "index.json");
    const index = JSON.parse(fs.readFileSync(file, "utf8"));
    index.active = "c-gone";
    fs.writeFileSync(file, JSON.stringify(index));
    fs.rmSync(`${file}.bak`, { force: true });
    const again = build();
    await again.init();
    expect(again.summary()).toMatchObject({ conversationId: saved.conversationId, title: "keep me", turns: 1, sessionId: saved.sessionId, readOnly: null });
    const after = JSON.parse(fs.readFileSync(file, "utf8"));
    expect(after.active).toBe(saved.conversationId);
    expect(Object.values(after.convs).some((c: any) => c.archived || c.readOnly)).toBe(false);
  });

  it("starts a new conversation if the pointer is lost and the only one cannot be continued", async () => {
    await chief.send("old one");
    await waitState("idle");
    await chief.stop(); // the save queued by the last state change must land before the files are edited
    const saved = chief.summary();
    const file = path.join(cfg.stateDir, "chief", "index.json");
    const index = JSON.parse(fs.readFileSync(file, "utf8"));
    index.active = "";
    index.convs[saved.conversationId].archived = true;
    index.convs[saved.conversationId].readOnly = "legacy";
    fs.writeFileSync(file, JSON.stringify(index));
    fs.writeFileSync(path.join(cfg.stateDir, "chief", "conv", saved.conversationId, "entry.json"), JSON.stringify(index.convs[saved.conversationId])); // the entry file is the source of truth
    fs.rmSync(`${file}.bak`, { force: true });
    const again = build();
    await again.init();
    const fresh = again.summary();
    expect(fresh.conversationId).not.toBe(saved.conversationId);
    expect(fresh).toMatchObject({ turns: 0, title: null, readOnly: null });
    const after = JSON.parse(fs.readFileSync(file, "utf8"));
    expect(after.active).toBe(fresh.conversationId);
    expect(after.convs[saved.conversationId]).toEqual(index.convs[saved.conversationId]);
  });

  it("resumes a parked conversation from its saved session", async () => {
    await chief.send("topic one");
    await waitState("idle");
    const first = chief.summary();
    await chief.reset();
    await chief.resume(first.conversationId);
    expect(chief.summary()).toMatchObject({ conversationId: first.conversationId, turns: 1, sessionId: first.sessionId, state: "idle", title: "topic one" });
    const before = starts.length;
    await chief.send("topic one, again");
    await waitState("idle");
    expect(starts).toHaveLength(before + 1);
    expect(starts[before].resume).toBe(first.sessionId);
    const log = (await chief.events(0, 100)).events;
    expect(log.some((e) => e.kind === "user" && e.text === "topic one")).toBe(true);
    expect(log.some((e) => e.kind === "user" && e.text === "topic one, again")).toBe(true);
  });

  it("stops a running turn when another conversation is resumed", async () => {
    await chief.send("first");
    await waitState("idle");
    const first = chief.summary().conversationId;
    await chief.reset();
    await chief.send("second");
    await chief.resume(first);
    expect(chief.summary()).toMatchObject({ conversationId: first, state: "idle" });
    expect(chief.conversations().find((c) => c.title === "second")).toMatchObject({ active: false });
  });

  it("lists conversations newest first and marks the active one", async () => {
    await chief.send("alpha");
    await waitState("idle");
    await chief.reset();
    await chief.send("beta");
    await waitState("idle");
    const rows = chief.conversations();
    expect(rows.map((r) => r.title)).toEqual(["beta", "alpha"]);
    expect(rows.map((r) => r.active)).toEqual([true, false]);
    expect(rows[1]).toMatchObject({ turns: 1, archived: false, readOnly: null, seededFrom: null });
  });

  it("renames, and a manual title is never replaced by the automatic one", async () => {
    const id = chief.summary().conversationId;
    await chief.patch(id, { title: "  Mine  " });
    await chief.send("something else entirely");
    await waitState("idle");
    expect(chief.summary().title).toBe("Mine");
    await expect(chief.patch(id, { title: "   " })).rejects.toMatchObject({ code: "invalid" });
    await expect(chief.patch(id, { title: "x".repeat(121) })).rejects.toMatchObject({ code: "invalid" });
  });

  it("archives a parked conversation, never the active one, and resuming restores it", async () => {
    await chief.send("alpha");
    await waitState("idle");
    const alpha = chief.summary().conversationId;
    await expect(chief.patch(alpha, { archived: true })).rejects.toMatchObject({ code: "state" });
    await chief.reset();
    await chief.patch(alpha, { archived: true });
    expect(chief.conversations().find((c) => c.id === alpha)).toMatchObject({ archived: true });
    await chief.resume(alpha);
    expect(chief.conversations().find((c) => c.id === alpha)).toMatchObject({ archived: false, active: true });
  });

  it("changes nothing when one part of a patch is invalid", async () => {
    await chief.send("alpha");
    await waitState("idle");
    const id = chief.summary().conversationId;
    await expect(chief.patch(id, { title: "New name", archived: true })).rejects.toMatchObject({ code: "state" });
    expect(chief.summary().title).toBe("alpha");
  });

  it("reads a parked conversation's log without activating it", async () => {
    await chief.send("parked words");
    await waitState("idle");
    const parked = chief.summary().conversationId;
    await chief.reset();
    const res = await chief.eventsFor(parked, 0, 100);
    expect(res.events.some((e) => e.kind === "user" && e.text === "parked words")).toBe(true);
    expect(res.lastSeq).toBe(res.events[res.events.length - 1].seq);
    expect(chief.summary().conversationId).not.toBe(parked);
    expect(await chief.eventsFor(chief.summary().conversationId, 0, 100)).toEqual(await chief.events(0, 100));
  });

  it("renaming or archiving does not count as activity", async () => {
    await chief.send("alpha");
    await waitState("idle");
    const alpha = chief.summary().conversationId;
    await chief.reset();
    const before = chief.conversations().map((c) => [c.id, c.updatedAt]);
    await new Promise((r) => setTimeout(r, 15));
    await chief.patch(alpha, { title: "Renamed" });
    await chief.patch(alpha, { archived: true });
    expect(chief.conversations().map((c) => [c.id, c.updatedAt])).toEqual(before);
  });

  it("never matches an id against Object.prototype members", async () => {
    for (const id of ["constructor", "toString", "__proto__"]) {
      await expect(chief.resume(id)).rejects.toMatchObject({ code: "not-found" });
      await expect(chief.patch(id, { title: "x" })).rejects.toMatchObject({ code: "not-found" });
      await expect(chief.eventsFor(id, 0, 10)).rejects.toMatchObject({ code: "not-found" });
    }
  });

  it("answers an unknown conversation with not-found", async () => {
    await expect(chief.resume("c-nope")).rejects.toMatchObject({ code: "not-found" });
    await expect(chief.patch("c-nope", { title: "x" })).rejects.toMatchObject({ code: "not-found" });
    await expect(chief.eventsFor("c-nope", 0, 10)).rejects.toMatchObject({ code: "not-found" });
  });

  it("will not resume a read-only conversation", async () => {
    const file = path.join(cfg.stateDir, "chief", "index.json");
    const index = JSON.parse(fs.readFileSync(file, "utf8"));
    index.convs["c-old"] = { id: "c-old", title: "old", titleManual: false, createdAt: "2026-10-01T00:00:00.000Z", updatedAt: "2026-10-01T00:00:00.000Z", turns: 1, costUsd: null, sessionId: null, archived: true, readOnly: "legacy", seededFrom: null };
    fs.writeFileSync(file, JSON.stringify(index));
    const again = build();
    await again.init();
    await expect(again.resume("c-old")).rejects.toMatchObject({ code: "state" });
  });

  it("keeps each conversation's session id in its own folder and survives losing every index file", async () => {
    await chief.send("alpha");
    await waitState("idle");
    await chief.stop();
    const alpha = chief.summary();
    const dir = path.join(cfg.stateDir, "chief");
    expect(alpha.sessionId).not.toBeNull();
    for (const f of fs.readdirSync(dir).filter((n) => n.startsWith("index.json"))) fs.rmSync(path.join(dir, f));
    const again = build();
    await again.init();
    expect(again.summary()).toMatchObject({ conversationId: alpha.conversationId, sessionId: alpha.sessionId, title: "alpha", readOnly: null });
  });

  it("brings back an entry the index lost", async () => {
    await chief.send("alpha");
    await waitState("idle");
    const alpha = chief.summary().conversationId;
    await chief.reset();
    const file = path.join(cfg.stateDir, "chief", "index.json");
    const index = JSON.parse(fs.readFileSync(file, "utf8"));
    delete index.convs[alpha];
    fs.writeFileSync(file, JSON.stringify(index));
    const again = build();
    await again.init();
    expect(again.conversations().find((c) => c.id === alpha)).toMatchObject({ title: "alpha", turns: 1 });
  });

  it("tells the owner when a change could not be saved, and leaves the title as it was", async () => {
    await chief.send("alpha");
    await waitState("idle");
    const alpha = chief.summary().conversationId;
    await chief.reset();
    const file = path.join(cfg.stateDir, "chief", "index.json");
    fs.rmSync(file);
    fs.mkdirSync(file); // a directory where the index should be: every save fails
    const log = vi.spyOn(console, "error").mockImplementation(() => undefined);
    await expect(chief.patch(alpha, { title: "Renamed" })).rejects.toMatchObject({ code: "state" });
    log.mockRestore();
    expect(chief.conversations().find((c) => c.id === alpha)?.title).toBe("alpha");
  });

  it("stores a typed title on one line", async () => {
    const id = chief.summary().conversationId;
    await chief.patch(id, { title: "  a\n\n\tb\u0000c  " });
    expect(chief.summary().title).toBe("a b c");
  });

  it("keeps a conversation seeded from another one when the owner switches away", async () => {
    const dir = path.join(cfg.stateDir, "chief");
    const id = chief.summary().conversationId;
    const file = path.join(dir, "index.json");
    const index = JSON.parse(fs.readFileSync(file, "utf8"));
    index.convs[id].seededFrom = "c-elsewhere";
    fs.writeFileSync(file, JSON.stringify(index));
    fs.writeFileSync(path.join(dir, "conv", id, "entry.json"), JSON.stringify(index.convs[id]));
    const again = build();
    await again.init();
    await again.reset();
    expect(again.conversations().find((c) => c.id === id)).toMatchObject({ seededFrom: "c-elsewhere" });
    expect(fs.existsSync(path.join(dir, "conv", id))).toBe(true);
  });

  it("removes empty leftovers of an interrupted switch when it starts", async () => {
    const dir = path.join(cfg.stateDir, "chief");
    const id = chief.summary().conversationId;
    const ghost = { ...JSON.parse(fs.readFileSync(path.join(dir, "conv", id, "entry.json"), "utf8")), id: "c-ghost", updatedAt: "2020-01-01T00:00:00.000Z" };
    fs.mkdirSync(path.join(dir, "conv", "c-ghost"), { recursive: true });
    fs.writeFileSync(path.join(dir, "conv", "c-ghost", "entry.json"), JSON.stringify(ghost));
    const again = build();
    await again.init();
    expect(again.summary().conversationId).toBe(id);
    expect(again.conversations().map((c) => c.id)).toEqual([id]);
    expect(fs.existsSync(path.join(dir, "conv", "c-ghost"))).toBe(false);
  });

  it("does not start on a read-only conversation that the index names as active", async () => {
    const file = path.join(cfg.stateDir, "chief", "index.json");
    const index = JSON.parse(fs.readFileSync(file, "utf8"));
    index.convs["c-old"] = { id: "c-old", title: "old", titleManual: false, createdAt: "2026-10-01T00:00:00.000Z", updatedAt: "2026-10-01T00:00:00.000Z", turns: 1, costUsd: null, sessionId: "s", archived: false, readOnly: "expired", seededFrom: null };
    index.active = "c-old";
    fs.writeFileSync(file, JSON.stringify(index));
    const again = build();
    await again.init();
    expect(again.summary().conversationId).not.toBe("c-old");
    expect(again.summary().readOnly).toBeNull();
    expect(again.conversations().find((c) => c.id === "c-old")).toMatchObject({ readOnly: "expired", active: false });
  });

  it("leaves no half-switched chief when the next journal cannot be opened", async () => {
    await chief.send("alpha");
    await waitState("idle");
    const alpha = chief.summary().conversationId;
    await chief.reset();
    const second = chief.summary().conversationId;
    // A plain file where alpha's folder should be makes opening its journal fail.
    fs.rmSync(path.join(cfg.stateDir, "chief", "conv", alpha), { recursive: true });
    fs.writeFileSync(path.join(cfg.stateDir, "chief", "conv", alpha), "x");
    await expect(chief.resume(alpha)).rejects.toThrow();
    expect(chief.summary().conversationId).toBe("");
    await expect(chief.send("lost?")).rejects.toMatchObject({ code: "state", message: expect.stringMatching(/no active conversation/i) });
    const saved = JSON.parse(fs.readFileSync(path.join(cfg.stateDir, "chief", "index.json"), "utf8"));
    expect(saved.active).toBe(""); // the index on disk agrees, and no longer lists the dropped conversation
    expect(Object.keys(saved.convs)).not.toContain(second);
    await chief.reset(); // a new conversation recovers
    expect(chief.summary().conversationId).not.toBe("");
  });

  it("re-reads a parked journal only when its file changed", async () => {
    await chief.send("alpha");
    await waitState("idle");
    const alpha = chief.summary().conversationId;
    await chief.reset();
    const first = await chief.eventsFor(alpha, 0, 100);
    expect(await chief.eventsFor(alpha, 0, 100)).toEqual(first);
    expect((await chief.eventsFor(alpha, first.lastSeq, 100)).events).toEqual([]);
  });
  it("names the conversation in every log answer, so a reader can tell when it changed", async () => {
    await chief.send("alpha");
    await waitState("idle");
    const alpha = chief.summary().conversationId;
    expect((await chief.events(0, 100)).conversationId).toBe(alpha);
    await chief.reset();
    const second = chief.summary().conversationId;
    expect((await chief.events(0, 100)).conversationId).toBe(second);
    expect((await chief.eventsFor(alpha, 0, 100)).conversationId).toBe(alpha);
  });

  it("labels a log read with the conversation it started on, even if a reset lands meanwhile", async () => {
    await chief.send("alpha");
    await waitState("idle");
    const alpha = chief.summary().conversationId;
    const pending = chief.events(0, 100);
    await chief.reset();
    expect((await pending).conversationId).toBe(alpha);
  });

  it("keeps the dropped empty conversation's folder when the switch could not be saved", async () => {
    const first = chief.summary().conversationId;
    const file = path.join(cfg.stateDir, "chief", "index.json");
    fs.rmSync(file);
    fs.mkdirSync(file); // every save fails
    const log = vi.spyOn(console, "error").mockImplementation(() => undefined);
    await expect(chief.reset()).rejects.toMatchObject({ code: "state" });
    log.mockRestore();
    expect(fs.existsSync(path.join(cfg.stateDir, "chief", "conv", first))).toBe(true);
  });

  it("clears the live counters of the conversation it left when the next one cannot be opened", async () => {
    await chief.send("alpha");
    await waitState("idle");
    const alpha = chief.summary().conversationId;
    await chief.reset();
    await chief.send("beta");
    await waitState("idle");
    const beta = chief.summary().conversationId;
    expect(chief.summary()).toMatchObject({ turns: 1 });
    fs.rmSync(path.join(cfg.stateDir, "chief", "conv", alpha), { recursive: true });
    fs.writeFileSync(path.join(cfg.stateDir, "chief", "conv", alpha), "x"); // alpha's journal cannot be opened
    await expect(chief.resume(alpha)).rejects.toThrow();
    expect(chief.summary()).toMatchObject({ conversationId: "", turns: 0, costUsd: null, sessionId: null, state: "idle" });
    expect(chief.conversations().find((c) => c.id === beta)).toMatchObject({ turns: 1, active: false }); // beta itself is safe
  });

  it("puts the old title back on disk when a rename could not be saved", async () => {
    await chief.send("alpha");
    await waitState("idle");
    const alpha = chief.summary().conversationId;
    await chief.reset();
    const file = path.join(cfg.stateDir, "chief", "index.json");
    fs.rmSync(file);
    fs.mkdirSync(file);
    const log = vi.spyOn(console, "error").mockImplementation(() => undefined);
    await expect(chief.patch(alpha, { title: "Renamed" })).rejects.toMatchObject({ code: "state" });
    log.mockRestore();
    const saved = JSON.parse(fs.readFileSync(path.join(cfg.stateDir, "chief", "conv", alpha, "entry.json"), "utf8"));
    expect(saved.title).toBe("alpha");
  });

  it("gives no title to a first message with nothing printable in it", async () => {
    await chief.send("\u0001\u0002");
    await waitState("idle");
    expect(chief.summary().title).toBeNull();
  });

  it("clears temp files a killed process left when it starts", async () => {
    await chief.stop();
    const dir = path.join(cfg.stateDir, "chief");
    const id = chief.summary().conversationId;
    fs.writeFileSync(path.join(dir, "index.json.99999.tmp"), "x");
    fs.writeFileSync(path.join(dir, "conv", id, "entry.json.99999.tmp"), "x");
    const again = build();
    await again.init();
    expect(fs.existsSync(path.join(dir, "index.json.99999.tmp"))).toBe(false);
    expect(fs.existsSync(path.join(dir, "conv", id, "entry.json.99999.tmp"))).toBe(false);
  });
});

describe("continuing in a new conversation", () => {
  it("continues in a new conversation: the first prompt carries what was said, once", async () => {
    await chief.send("we agreed on Tuesday");
    await waitState("idle");
    const src = chief.summary().conversationId;
    await chief.continueFrom(src);
    expect(chief.summary()).toMatchObject({ turns: 0, sessionId: null, title: null });
    expect(chief.conversations().find((c) => c.active)).toMatchObject({ seededFrom: src });
    const before = starts.length;
    await chief.send("what did we agree?");
    await waitState("idle");
    expect(starts[before].prompt).toContain("we agreed on Tuesday");
    expect(starts[before].prompt).toContain("The owner says:\nwhat did we agree?");
    expect(starts[before].resume).toBeUndefined();
    await chief.send("and next?");
    await waitState("idle");
    expect(starts).toHaveLength(before + 1); // later messages go to the live session, not a new prompt
  });

  it("refuses to continue from an unknown or empty conversation", async () => {
    await expect(chief.continueFrom("c-nope")).rejects.toMatchObject({ code: "not-found" });
    await expect(chief.continueFrom(chief.summary().conversationId)).rejects.toMatchObject({ code: "invalid" });
  });
});

describe("a conversation whose session is gone", () => {
  const gone = { transcriptExists: async () => false };

  it("is read-only after a restart when its saved transcript has vanished", async () => {
    await chief.send("remember this");
    await waitState("idle");
    const id = chief.summary().conversationId;
    const again = build({ stepMs: 0 }, undefined, gone);
    await again.init();
    expect(again.summary()).toMatchObject({ conversationId: id, readOnly: "expired", sessionId: null });
    await expect(again.send("hello?")).rejects.toMatchObject({ code: "state" });
    expect((await again.events(0, 100)).events.filter((e) => e.kind === "user")).toHaveLength(1); // the refused message was not journalled
  });

  it("refuses to resume a parked conversation whose transcript vanished, and says so in the list", async () => {
    await chief.send("topic");
    await waitState("idle");
    const first = chief.summary().conversationId;
    await chief.reset();
    const later = build({ stepMs: 0 }, undefined, gone);
    await later.init();
    await expect(later.resume(first)).rejects.toMatchObject({ code: "state" });
    expect(later.conversations().find((c) => c.id === first)).toMatchObject({ readOnly: "expired" });
    // and it is remembered across another restart without probing again
    const third = build();
    await third.init();
    expect(third.conversations().find((c) => c.id === first)).toMatchObject({ readOnly: "expired" });
  });

  it("turns a resume that fails before saying anything into a read-only conversation, not a failed one", async () => {
    await chief.send("topic");
    await waitState("idle");
    const id = chief.summary().conversationId;
    const failing: AgentDriver = {
      start: () => ({
        events: (async function* () { throw new Error("No conversation found with session ID"); })(),
        send() {}, interrupt: async () => {}, close() {},
      }),
    };
    const again = build({}, failing);
    await again.init();
    await again.send("are you there?");
    await until(() => (again.summary().readOnly && again.summary().state === "idle") || undefined);
    expect(again.summary()).toMatchObject({ conversationId: id, state: "idle", readOnly: "expired", sessionId: null, error: null });
    expect((await again.events(0, 100)).events.some((e) => e.kind === "error")).toBe(true);
  });

  it("handles the way the real SDK reports a missing session: a failed result, then an error", async () => {
    await chief.send("topic");
    await waitState("idle");
    const id = chief.summary().conversationId;
    const real: AgentDriver = {
      start: () => ({
        events: (async function* () {
          yield { body: { kind: "result" as const, ok: false, text: "error_during_execution", turns: 0, costUsd: null } };
          throw new Error("Claude Code returned an error result: No conversation found with session ID: s-1");
        })(),
        send() {}, interrupt: async () => {}, close() {},
      }),
    };
    const again = build({}, real);
    await again.init();
    await again.send("are you there?");
    await until(() => (again.summary().readOnly && again.summary().state === "idle") || undefined);
    expect(again.summary()).toMatchObject({ conversationId: id, readOnly: "expired", turns: 1, error: null });
    const kinds = (await again.events(0, 100)).events.map((e) => e.kind);
    expect(kinds.filter((k) => k === "result")).toHaveLength(1); // only the real turn from before; the failed attempt is not one
  });

  it("does not expire a conversation because a resume failed for another reason", async () => {
    await chief.send("topic");
    await waitState("idle");
    const flaky: AgentDriver = {
      start: () => ({ events: (async function* () { throw new Error("socket hang up"); })(), send() {}, interrupt: async () => {}, close() {} }),
    };
    const again = build({}, flaky);
    await again.init();
    const log = vi.spyOn(console, "error").mockImplementation(() => undefined);
    await again.send("hello?");
    await until(() => again.summary().state === "failed" || undefined);
    log.mockRestore();
    expect(again.summary().readOnly).toBeNull();
    expect(again.summary().sessionId).not.toBeNull(); // still resumable once the network is back
  });

  it("treats a resume that quietly starts a different session as a lost session", async () => {
    await chief.send("topic");
    await waitState("idle");
    const stray: AgentDriver = {
      start: () => ({
        events: (async function* () { yield { sessionId: "a-brand-new-session" }; await new Promise(() => undefined); })(),
        send() {}, interrupt: async () => {}, close() {},
      }),
    };
    const again = build({}, stray);
    await again.init();
    await again.send("are you there?");
    await until(() => (again.summary().readOnly && again.summary().state === "idle") || undefined);
    expect(again.summary()).toMatchObject({ readOnly: "expired", sessionId: null, state: "idle" });
  });

  it("does not treat an ordinary failure of a brand-new conversation as expiry", async () => {
    const failing: AgentDriver = {
      start: () => ({ events: (async function* () { throw new Error("boom"); })(), send() {}, interrupt: async () => {}, close() {} }),
    };
    chief = build({}, failing);
    await chief.init();
    await chief.reset();
    await chief.send("first words");
    await until(() => chief.summary().state === "failed" || undefined);
    expect(chief.summary().readOnly).toBeNull();
  });
});
