import { describe, it, expect, vi } from "vitest";
import { buildServer } from "../server/wiring";
import { EventBus } from "../server/events";
import { FakeGh } from "./helpers/fakeGh";
import type { Config } from "../server/config";

const TOKEN = "t".repeat(64);
const OWN = "http://127.0.0.1:4317";
const COOKIE = { Cookie: `dash_token=${TOKEN}` };
const BEARER = { Authorization: `Bearer ${TOKEN}` };

function server() {
  const gh = new FakeGh();
  const start = vi.fn(() => true);
  const tests = { state: () => ({ status: "idle" }) as never, start, stop: () => Promise.resolve() };
  const kanban = {
    board: vi.fn(async () => ({}) as never),
    sync: vi.fn(async () => {}),
    addTask: vi.fn(async () => {}),
    setStatus: vi.fn(async () => {}),
    fileChanged: vi.fn(),
    poll: vi.fn(),
    dispose: vi.fn(),
  };
  const agents = {
    list: vi.fn(() => []),
    get: vi.fn(() => ({}) as never),
    spawn: vi.fn(async () => ({}) as never),
    message: vi.fn(async () => {}),
    stop: vi.fn(async () => {}),
    resume: vi.fn(async () => {}),
    events: vi.fn(async () => ({ events: [], lastSeq: 0 })),
    init: vi.fn(async () => {}),
    stopAll: vi.fn(async () => {}),
  };
  const approvals = { pending: vi.fn(() => []), decide: vi.fn(), request: vi.fn(), cancelAgent: vi.fn(), count: vi.fn(() => 0), countFor: vi.fn(() => 0) };
  const chief = { summary: vi.fn(() => ({}) as never), send: vi.fn(async () => {}), stop: vi.fn(async () => {}), reset: vi.fn(async () => {}), events: vi.fn(async () => ({ events: [], lastSeq: 0 })), conversations: vi.fn(() => []), resume: vi.fn(async () => {}), patch: vi.fn(async () => {}), eventsFor: vi.fn(async () => ({ events: [], lastSeq: 0 })), init: vi.fn(async () => {}) };
  const terminal = { enabled: true, open: vi.fn(() => ({ id: "t" })), attach: vi.fn(), input: vi.fn(), resize: vi.fn(), close: vi.fn(async () => {}), closeAll: vi.fn(async () => {}), count: vi.fn(() => 0) };
  const app = buildServer({
    config: { webDist: "/nonexistent", repoRoot: "/tmp" } as Config,
    token: TOKEN,
    bus: new EventBus(),
    gh,
    tests,
    kanban,
    runner: async () => ({ stdout: "x\n", stderr: "", code: 0 }),
    runs: { scan: async () => ({ runs: [], scannedAt: "", checkpointsDirExists: false, truncated: false, error: null }) },
    agents,
    approvals,
    chief,
    terminal: terminal as never,
  });
  const effects = () => ({
    gh: gh.writes.length,
    start: start.mock.calls.length,
    kanban: kanban.addTask.mock.calls.length + kanban.setStatus.mock.calls.length + kanban.sync.mock.calls.length,
    agents: agents.spawn.mock.calls.length + agents.message.mock.calls.length + agents.stop.mock.calls.length + agents.resume.mock.calls.length,
    approvals: approvals.decide.mock.calls.length,
    chief: chief.send.mock.calls.length + chief.stop.mock.calls.length + chief.reset.mock.calls.length,
    terminal: terminal.open.mock.calls.length + terminal.input.mock.calls.length + terminal.resize.mock.calls.length + terminal.close.mock.calls.length,
  });
  return { app, effects };
}

const MUTATING: [string, string, unknown][] = [
  ["POST", "/api/issues", { title: "x", body: "y" }],
  ["POST", "/api/tests/run", undefined],
  ["POST", "/api/kanban/tasks", { title: "x" }],
  ["PATCH", "/api/kanban/tasks/1", { status: "done" }],
  ["POST", "/api/kanban/sync", undefined],
  ["POST", "/api/agents", { name: "abc-def", task: "t" }],
  ["POST", "/api/agents/abc-def/message", { text: "hi" }],
  ["POST", "/api/agents/abc-def/stop", undefined],
  ["POST", "/api/agents/abc-def/resume", undefined],
  ["POST", "/api/approvals/abc/decision", { decision: "allow" }],
  ["POST", "/api/chief/message", { text: "hi" }],
  ["POST", "/api/chief/stop", undefined],
  ["POST", "/api/chief/reset", undefined],
  ["POST", "/api/terminal/open", { cols: 80, rows: 24 }],
  ["POST", "/api/terminal/abc/input", { data: "ls\r" }],
  ["POST", "/api/terminal/abc/resize", { cols: 80, rows: 24 }],
  ["DELETE", "/api/terminal/abc", undefined],
];

const send = (app: ReturnType<typeof server>["app"], method: string, url: string, body: unknown, headers: Record<string, string>) =>
  app.request(`${OWN}${url}`, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
const json = { "Content-Type": "application/json" };

describe.each(MUTATING)("%s %s CSRF protection", (method, url, body) => {
  it("refuses a forged cross-port request that carries a valid cookie, and does nothing", async () => {
    const { app, effects } = server();
    for (const headers of [
      { ...COOKIE, ...json, Origin: "http://127.0.0.1:8888" },
      { ...COOKIE, "Content-Type": "text/plain", Origin: "http://127.0.0.1:8888" },
      { ...COOKIE, ...json, Origin: "null" },
      { ...COOKIE, ...json, "Sec-Fetch-Site": "same-site" },
      { ...COOKIE, ...json, "Sec-Fetch-Site": "cross-site" },
    ]) {
      const res = await send(app, method, url, body, headers);
      expect(res.status, JSON.stringify(headers)).toBe(403);
    }
    expect(effects()).toEqual({ gh: 0, start: 0, kanban: 0, agents: 0, approvals: 0, chief: 0, terminal: 0 });
  });

  it("requires Content-Type: application/json (415), even for body-less routes", async () => {
    const { app, effects } = server();
    for (const ct of [undefined, "text/plain", "application/x-www-form-urlencoded", "text/plain;application/json"]) {
      const headers: Record<string, string> = { ...COOKIE, Origin: OWN };
      if (ct) headers["Content-Type"] = ct;
      const res = await send(app, method, url, body, headers);
      expect(res.status, String(ct)).toBe(415);
    }
    expect(effects()).toEqual({ gh: 0, start: 0, kanban: 0, agents: 0, approvals: 0, chief: 0, terminal: 0 });
  });

  it("accepts same-origin JSON with the cookie, with or without Origin/Sec-Fetch-Site", async () => {
    for (const extra of [{ Origin: OWN }, { "Sec-Fetch-Site": "same-origin" }, { "Sec-Fetch-Site": "none" }, {}] as Record<string, string>[]) {
      const { app } = server();
      const res = await send(app, method, url, body, { ...COOKIE, ...json, ...extra });
      expect(res.status, JSON.stringify(extra)).toBeLessThan(400);
    }
    const { app } = server();
    const ok = await send(app, method, url, body, { ...COOKIE, "Content-Type": "application/json; charset=utf-8", Origin: OWN });
    expect(ok.status).toBeLessThan(400);
  });

  it("exempts Bearer-authenticated requests from the Origin check but still wants JSON", async () => {
    const { app } = server();
    const res = await send(app, method, url, body, { ...BEARER, ...json, Origin: "http://127.0.0.1:8888" });
    expect(res.status).toBeLessThan(400);
    const bad = await send(app, method, url, body, { ...BEARER, "Content-Type": "text/plain" });
    expect(bad.status).toBe(415);
  });

  it("still answers 401 first without credentials", async () => {
    const { app } = server();
    expect((await send(app, method, url, body, { ...json, Origin: "http://evil" })).status).toBe(401);
  });
});

describe("safe methods", () => {
  it("are not subject to the Origin or Content-Type rules", async () => {
    const { app } = server();
    const res = await app.request(`${OWN}/api/health`, { headers: { ...COOKIE, Origin: "http://127.0.0.1:8888", "Sec-Fetch-Site": "cross-site" } });
    expect(res.status).toBe(200);
  });
});
