import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { EventEmitter } from "node:events";
import { PassThrough } from "node:stream";
import { buildServer } from "../server/wiring";
import { EventBus } from "../server/events";
import { createTerminal, type Terminal } from "../server/terminal/terminal";
import { loadConfig, type Config } from "../server/config";

const TOKEN = "t".repeat(64);
const OWN = "http://127.0.0.1:4317";
const BEARER = { Authorization: `Bearer ${TOKEN}`, "Content-Type": "application/json" };
const COOKIE = { Cookie: `dash_token=${TOKEN}`, "Content-Type": "application/json" };

class FakeChild extends EventEmitter {
  stdin = new PassThrough();
  stdout = new PassThrough();
  pid = 777;
  written: Buffer[] = [];
  constructor() {
    super();
    this.stdin.on("data", (d: Buffer) => this.written.push(d));
  }
  exit() {
    this.stdout.end();
    this.emit("exit", 0, null);
  }
}

let children: FakeChild[];
let term: Terminal;
let app: ReturnType<typeof buildServer>;

function makeApp(over: Partial<Config> = {}) {
  const cfg = { ...loadConfig({ DASH_REPO_ROOT: "/r/repo" }), webDist: "/nonexistent", ...over };
  term = createTerminal({
    cfg,
    env: { PATH: "/bin" },
    spawn: () => {
      const c = new FakeChild();
      children.push(c);
      return c as never;
    },
    kill: () => undefined,
    killAfterMs: 10,
  });
  const none = { list: () => [], get: () => undefined, spawn: vi.fn(), message: vi.fn(), stop: vi.fn(), resume: vi.fn(), events: vi.fn(), init: vi.fn(), stopAll: vi.fn() };
  return buildServer({
    config: cfg,
    token: TOKEN,
    bus: new EventBus(),
    gh: { json: vi.fn(), text: vi.fn() } as never,
    tests: { state: () => ({ status: "idle" }) as never, start: () => true, stop: () => Promise.resolve() },
    kanban: { board: vi.fn(), sync: vi.fn(), addTask: vi.fn(), setStatus: vi.fn(), fileChanged: vi.fn(), poll: vi.fn(), dispose: vi.fn() } as never,
    runner: async () => ({ stdout: "x\n", stderr: "", code: 0 }),
    runs: { scan: async () => ({ runs: [], scannedAt: "", checkpointsDirExists: false, truncated: false, error: null }) },
    agents: none as never,
    approvals: { pending: () => [], decide: vi.fn(), request: vi.fn(), cancelAgent: vi.fn(), count: () => 0, countFor: () => 0 } as never,
    chief: { summary: () => ({}) as never, send: vi.fn(), stop: vi.fn(), reset: vi.fn(), events: vi.fn(), init: vi.fn() } as never,
    terminal: term,
  });
}

beforeEach(() => {
  children = [];
  app = makeApp();
});
afterEach(async () => {
  await term.closeAll(200);
});

const post = (url: string, body?: unknown, headers: Record<string, string> = BEARER) =>
  app.request(OWN + url, { method: "POST", headers, body: body === undefined ? undefined : JSON.stringify(body) });
const get = (url: string, headers: Record<string, string> = BEARER) => app.request(OWN + url, { headers });
const open = async (cols = 100, rows = 30): Promise<string> => ((await (await post("/api/terminal/open", { cols, rows })).json()) as { id: string }).id;

describe("status", () => {
  it("reports whether the terminal is on and how many streams may be open", async () => {
    expect(await (await get("/api/terminal")).json()).toEqual({ enabled: true, max: 2, open: 0 });
    await open();
    expect(((await (await get("/api/terminal")).json()) as { open: number }).open).toBe(1);
  });
  it("says so when it is off", async () => {
    app = makeApp({ terminalEnabled: false });
    expect(await (await get("/api/terminal")).json()).toMatchObject({ enabled: false });
    expect((await post("/api/terminal/open", { cols: 80, rows: 24 })).status).toBe(403);
  });
});

describe("open", () => {
  it("returns an id", async () => {
    const res = await post("/api/terminal/open", { cols: 100, rows: 30 });
    expect(res.status).toBe(201);
    expect(((await res.json()) as { id: string }).id).toMatch(/^[0-9a-f-]{36}$/);
  });
  it.each([[{}], [{ cols: "80", rows: 24 }], [{ cols: 80 }], [{ cols: 0, rows: 24 }], [{ cols: 80, rows: 9999 }], [[1, 2]]])("rejects %j", async (body) => {
    expect((await post("/api/terminal/open", body)).status).toBe(400);
    expect(term.count()).toBe(0);
  });
  it("answers 409 at the stream limit", async () => {
    await open();
    await open();
    expect((await post("/api/terminal/open", { cols: 80, rows: 24 })).status).toBe(409);
  });
});

describe("stream", () => {
  it("streams the shell's bytes with no caching, once per id", async () => {
    const id = await open();
    const res = await get(`/api/terminal/${id}/stream`);
    expect(res.status).toBe(200);
    expect(res.headers.get("Content-Type")).toBe("application/octet-stream");
    expect(res.headers.get("Cache-Control")).toBe("no-store");
    const reader = (res.body as ReadableStream<Uint8Array>).getReader();
    children[0].stdout.write(Buffer.from([0x68, 0x69, 0xc3]));
    children[0].stdout.write(Buffer.from([0xa9]));
    let got = Buffer.alloc(0);
    while (got.length < 4) got = Buffer.concat([got, Buffer.from((await reader.read()).value as Uint8Array)]);
    expect(got.toString("utf8")).toBe("hié");
    expect((await get(`/api/terminal/${id}/stream`)).status).toBe(409);
    children[0].exit();
    expect((await reader.read()).done).toBe(true);
  });

  it("closes the shell when the browser goes away", async () => {
    const id = await open();
    const ac = new AbortController();
    const res = await app.request(`${OWN}/api/terminal/${id}/stream`, { headers: BEARER, signal: ac.signal });
    const reader = (res.body as ReadableStream<Uint8Array>).getReader();
    children[0].stdout.write("x");
    await reader.read();
    ac.abort();
    await reader.cancel().catch(() => undefined);
    await new Promise((r) => setTimeout(r, 80));
    expect(children[0].stdin.writableEnded).toBe(true);
    expect(term.count()).toBe(0);
  });

  it("is refused for another site even with the cookie, because attaching has side effects", async () => {
    const id = await open();
    for (const extra of [{ Origin: "https://evil.example" }, { "Sec-Fetch-Site": "cross-site" }, { "Sec-Fetch-Site": "same-site" }] as Record<string, string>[]) {
      const res = await get(`/api/terminal/${id}/stream`, { Cookie: COOKIE.Cookie, ...extra });
      expect(res.status, JSON.stringify(extra)).toBe(403);
    }
    expect(children).toHaveLength(0);
    expect((await get(`/api/terminal/${id}/stream`, { Cookie: COOKIE.Cookie, Origin: OWN, "Sec-Fetch-Site": "same-origin" })).status).toBe(200);
  });

  it("404s an unknown id", async () => {
    expect((await get("/api/terminal/nope/stream")).status).toBe(404);
  });
});

describe("input, resize and close", () => {
  it("passes keystrokes and resizes through, in order", async () => {
    const id = await open();
    await get(`/api/terminal/${id}/stream`);
    expect((await post(`/api/terminal/${id}/input`, { data: "ls\r" })).status).toBe(200);
    expect((await post(`/api/terminal/${id}/resize`, { cols: 120, rows: 40 })).status).toBe(200);
    const all = Buffer.concat(children[0].written);
    expect(all.toString("latin1", 0, 1)).toBe("d");
    expect(all.subarray(5, 8).toString()).toBe("ls\r");
    expect(all.toString("latin1", 8, 9)).toBe("r");
  });
  it("validates bodies", async () => {
    const id = await open();
    await get(`/api/terminal/${id}/stream`);
    expect((await post(`/api/terminal/${id}/input`, { data: 5 })).status).toBe(400);
    expect((await post(`/api/terminal/${id}/input`, { data: "" })).status).toBe(400);
    expect((await post(`/api/terminal/${id}/input`, { data: "x".repeat(70_000) })).status).toBe(400);
    expect((await post(`/api/terminal/${id}/input`, {})).status).toBe(400);
    expect((await post(`/api/terminal/${id}/resize`, { cols: "a", rows: 1 })).status).toBe(400);
    expect((await post(`/api/terminal/${id}/resize`, { cols: 0, rows: 5 })).status).toBe(400);
  });
  it("404s unknown ids and refuses input before the stream is attached", async () => {
    expect((await post("/api/terminal/nope/input", { data: "x" })).status).toBe(404);
    const id = await open();
    expect((await post(`/api/terminal/${id}/input`, { data: "x" })).status).toBe(409);
  });
  it("DELETE closes the stream", async () => {
    const id = await open();
    await get(`/api/terminal/${id}/stream`);
    const res = await app.request(`${OWN}/api/terminal/${id}`, { method: "DELETE", headers: BEARER });
    expect(res.status).toBe(200);
    expect(term.count()).toBe(0);
    expect(children[0].stdin.writableEnded).toBe(true);
  });
});

describe("access", () => {
  it.each([
    ["GET", "/api/terminal"],
    ["POST", "/api/terminal/open"],
    ["GET", "/api/terminal/x/stream"],
    ["POST", "/api/terminal/x/input"],
    ["POST", "/api/terminal/x/resize"],
    ["DELETE", "/api/terminal/x"],
  ])("%s %s without a token is 401", async (method, url) => {
    const res = await app.request(OWN + url, { method, headers: { "Content-Type": "application/json" } });
    expect(res.status).toBe(401);
    expect(term.count()).toBe(0);
  });

  it("refuses a cross-origin open, input or close even with the cookie", async () => {
    const id = await open();
    await get(`/api/terminal/${id}/stream`);
    const evil = { ...COOKIE, Origin: "https://evil.example" };
    expect((await post("/api/terminal/open", { cols: 80, rows: 24 }, evil)).status).toBe(403);
    expect((await post(`/api/terminal/${id}/input`, { data: "rm -rf /\r" }, evil)).status).toBe(403);
    expect((await app.request(`${OWN}/api/terminal/${id}`, { method: "DELETE", headers: evil })).status).toBe(403);
    expect(children[0].written).toHaveLength(0);
  });
});
