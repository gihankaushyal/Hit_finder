import { describe, it, expect, vi, afterEach } from "vitest";
import { Hono } from "hono";
import { FakeGh } from "./helpers/fakeGh";
import { EventBus } from "../server/events";
import { createKanbanService, NotImportedError } from "../server/lib/kanbanService";
import { kanbanRoutes } from "../server/routes/kanban";
import { buildServer } from "../server/wiring";
import type { SyncDeps, SyncState } from "../server/lib/kanbanSync";
import type { Board, DashEvent } from "../shared/types";
import type { Config } from "../server/config";

const FILE = ["## Open decisions (need a call)", "", "- [ ] **Open thing**", "", "## In progress (Phase 5)", "", "- [ ] Working on it", ""].join("\n");

function setup(opts: { file?: string | null; imported?: boolean; gh?: FakeGh } = {}) {
  const gh = opts.gh ?? new FakeGh();
  const h = {
    file: (opts.file === undefined ? FILE : opts.file) as string | null,
    state: { labelsEnsured: false, items: {} } as SyncState,
    fileWrites: 0,
    imported: opts.imported ?? true,
  };
  const sync: SyncDeps = {
    gh,
    readFile: () => h.file,
    writeFile: (t) => {
      h.file = t;
      h.fileWrites++;
    },
    loadState: () => structuredClone(h.state),
    saveState: (s) => {
      h.state = structuredClone(s);
    },
    sleep: async () => {},
  };
  const bus = new EventBus();
  const events: DashEvent[] = [];
  bus.subscribe((e) => events.push(e));
  const service = createKanbanService({ gh, sync, bus, imported: () => h.imported });
  const api = new Hono();
  kanbanRoutes(api, { service });
  const req = (method: string, url: string, body?: unknown, raw = false) =>
    api.request(url, {
      method,
      headers: { "Content-Type": "application/json" },
      body: body === undefined ? undefined : raw ? (body as string) : JSON.stringify(body),
    });
  const board = async (): Promise<Board> => (await req("GET", "/kanban")).json();
  return { gh, h, bus, events, service, api, req, board };
}

afterEach(() => {
  vi.useRealTimers();
});

describe("GET /kanban", () => {
  it("groups issues by status label and closed state; Done is newest first", async () => {
    const { gh, board } = setup();
    gh.seed({ title: "t1", labels: ["task", "status:todo", "kind:decision"] });
    gh.seed({ title: "t2", labels: ["task", "status:in-progress"] });
    gh.seed({ title: "t3", labels: ["task", "status:blocked"] });
    gh.seed({ title: "t4", labels: ["task"] });
    gh.seed({ title: "old", state: "CLOSED", labels: ["task", "status:todo"], updatedAt: "2026-10-01T00:00:00Z" });
    gh.seed({ title: "new", state: "CLOSED", labels: ["task"], updatedAt: "2026-10-05T00:00:00Z" });
    const b = await board();
    expect(b.imported).toBe(true);
    expect(b.columns.todo.map((t) => t.title)).toEqual(["t1", "t4"]);
    expect(b.columns.todo[0].kind).toBe("decision");
    expect(b.columns["in-progress"].map((t) => t.title)).toEqual(["t2"]);
    expect(b.columns.blocked.map((t) => t.title)).toEqual(["t3"]);
    expect(b.columns.done.map((t) => t.title)).toEqual(["new", "old"]);
  });

  it("marks a task inFile=false when the file no longer references it", async () => {
    const { gh, h, req, board } = setup({ file: "# nothing\n" });
    gh.seed({ title: "gone", labels: ["task", "status:todo"] });
    h.state.items["1"] = { checked: false, closed: false, bodyHash: "x" };
    expect((await req("POST", "/kanban/sync")).status).toBe(200);
    const b = await board();
    expect(b.columns.todo[0].inFile).toBe(false);
  });
});

describe("POST /kanban/tasks", () => {
  it("creates a task issue with task and status:todo, shows it in Todo and in the Inbox section", async () => {
    const { gh, h, req, board } = setup();
    const res = await req("POST", "/kanban/tasks", { title: "Write docs", body: "details" });
    expect(res.status).toBe(201);
    const create = gh.writes.find((w) => w[0] === "issue" && w[1] === "create")!;
    expect(create).toContain("--title=Write docs");
    expect(create).toContain("--label=task");
    expect(create).toContain("--label=status:todo");
    // the first mutation also links the existing file items, so find ours by title
    const b = await board();
    expect(b.columns.todo.map((t) => t.title)).toContain("Write docs");
    expect(h.file).toContain("Write docs");
    expect(h.file).toMatch(/## Inbox \(added via dashboard\)/);
  });

  it.each([
    ["missing title", {}],
    ["empty title", { title: "   " }],
    ["title too long", { title: "x".repeat(201) }],
    ["non-string title", { title: 5 }],
    ["NUL in title", { title: "a\0b" }],
    ["non-string body", { title: "ok", body: 3 }],
    ["array body", []],
  ])("rejects %s with 400 and no gh call", async (_n, payload) => {
    const { gh, req } = setup();
    const res = await req("POST", "/kanban/tasks", payload);
    expect(res.status).toBe(400);
    expect(gh.writes).toEqual([]);
  });

  it("rejects invalid JSON and oversized bodies with 400", async () => {
    const { gh, req } = setup();
    expect((await req("POST", "/kanban/tasks", "{nope", true)).status).toBe(400);
    expect((await req("POST", "/kanban/tasks", { title: "ok", body: "y".repeat(70 * 1024) })).status).toBe(400);
    expect(gh.writes).toEqual([]);
  });
});

describe("PATCH /kanban/tasks/:n", () => {
  const withSyncedItem = async () => {
    const s = setup({ file: "## Open decisions\n\n- [ ] one\n" });
    expect((await s.req("POST", "/kanban/sync")).status).toBe(200); // links the item as issue #1
    s.gh.writes.length = 0;
    return s;
  };

  it("moving to in-progress swaps the status labels", async () => {
    const { gh, req } = await withSyncedItem();
    const res = await req("PATCH", "/kanban/tasks/1", { status: "in-progress" });
    expect(res.status).toBe(200);
    const edit = gh.writes.find((w) => w[1] === "edit")!;
    expect(edit).toEqual(["issue", "edit", "1", "--add-label=status:in-progress", "--remove-label=status:todo"]);
    expect(gh.issues.get(1)!.labels).toContain("status:in-progress");
    expect(gh.issues.get(1)!.labels).not.toContain("status:todo");
  });

  it("done closes the issue and ticks the checkbox", async () => {
    const { gh, h, req } = await withSyncedItem();
    expect((await req("PATCH", "/kanban/tasks/1", { status: "done" })).status).toBe(200);
    expect(gh.issues.get(1)!.state).toBe("CLOSED");
    expect(h.file).toContain("- [x] one");
  });

  it("moving a closed task to todo reopens it and unticks the checkbox", async () => {
    const { gh, h, req } = await withSyncedItem();
    await req("PATCH", "/kanban/tasks/1", { status: "done" });
    gh.writes.length = 0;
    expect((await req("PATCH", "/kanban/tasks/1", { status: "todo" })).status).toBe(200);
    expect(gh.writes.some((w) => w[1] === "reopen" && w[2] === "1")).toBe(true);
    expect(gh.issues.get(1)!.state).toBe("OPEN");
    expect(h.file).toContain("- [ ] one");
  });

  it.each([
    ["/kanban/tasks/1", { status: "nope" }],
    ["/kanban/tasks/1", {}],
    ["/kanban/tasks/1", { status: 3 }],
    ["/kanban/tasks/0", { status: "todo" }],
    ["/kanban/tasks/-1", { status: "todo" }],
    ["/kanban/tasks/1.5", { status: "todo" }],
    ["/kanban/tasks/abc", { status: "todo" }],
    ["/kanban/tasks/99999999999999999999", { status: "todo" }],
  ])("rejects %s %j with 400 and no gh call", async (url, payload) => {
    const { gh, req } = setup();
    gh.seed({ title: "a" });
    const res = await req("PATCH", url, payload);
    expect(res.status).toBe(400);
    expect(gh.writes).toEqual([]);
  });

  it("returns 404 for an issue that is not a task", async () => {
    const { req } = setup();
    expect((await req("PATCH", "/kanban/tasks/42", { status: "done" })).status).toBe(404);
  });
});

describe("sync failures and events", () => {
  it("a sync failure after a mutation shows as syncError, never raw text, and the route still succeeds", async () => {
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    const { gh, req, board } = setup();
    const orig = gh.text.bind(gh);
    gh.text = async (args) => {
      const out = await orig(args);
      if (args[1] === "create") gh.failNextJson = true; // the sync that follows fails
      return out;
    };
    const res = await req("POST", "/kanban/tasks", { title: "x" });
    expect(res.status).toBe(201);
    const b = await board();
    expect(b.syncError).toBeTruthy();
    expect(b.syncError).not.toContain("gh list failed");
    spy.mockRestore();
  });

  it("a mutation emits exactly one kanban invalidate; a failed one emits none", async () => {
    const { gh, events, req } = setup();
    await req("POST", "/kanban/tasks", { title: "x" });
    expect(events.filter((e) => e.type === "invalidate" && e.resource === "kanban")).toHaveLength(1);
    events.length = 0;
    await req("PATCH", "/kanban/tasks/1", { status: "nope" });
    await req("PATCH", "/kanban/tasks/77", { status: "done" });
    expect(events).toEqual([]);
    void gh;
  });

  it("a gh failure on a mutation is a 502 with a fixed message", async () => {
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    const { gh, req } = setup();
    gh.failCreateAt = 1;
    const res = await req("POST", "/kanban/tasks", { title: "x" });
    expect(res.status).toBe(500);
    expect(JSON.stringify(await res.json())).not.toContain("gh create failed");
    spy.mockRestore();
  });
});

describe("before the first import", () => {
  it("serves the board straight from the markdown and makes no gh call", async () => {
    const { gh, req, board } = setup({ imported: false, file: FILE + "\n- [x] done thing\n" });
    const res = await req("GET", "/kanban");
    expect(res.status).toBe(200);
    const b = await board();
    expect(b.imported).toBe(false);
    expect(b.columns.todo.map((t) => t.title)).toEqual(["Open thing"]);
    expect(b.columns["in-progress"].map((t) => t.title)).toEqual(["Working on it"]);
    expect(b.columns.done.map((t) => t.title)).toEqual(["done thing"]);
    expect(new Set(Object.values(b.columns).flat().map((t) => t.number)).size).toBe(3); // unique ids
    expect(gh.jsonCalls).toEqual([]);
    expect(gh.writes).toEqual([]);
  });

  it("a missing file gives an empty board", async () => {
    const { board } = setup({ imported: false, file: null });
    const b = await board();
    expect(Object.values(b.columns).flat()).toEqual([]);
  });

  it.each([
    ["POST", "/kanban/tasks", { title: "x" }],
    ["PATCH", "/kanban/tasks/1", { status: "done" }],
    ["POST", "/kanban/sync", undefined],
  ])("%s %s returns 409 telling the user to run the CLI import, and does nothing", async (method, url, body) => {
    const { gh, h, req } = setup({ imported: false });
    const res = await req(method, url, body);
    expect(res.status).toBe(409);
    expect(((await res.json()) as { error: string }).error).toMatch(/npm run kanban:sync/);
    expect(gh.writes).toEqual([]);
    expect(gh.jsonCalls).toEqual([]);
    expect(h.fileWrites).toBe(0);
  });

  it("the service rejects sync() with NotImportedError", async () => {
    const { service } = setup({ imported: false });
    await expect(service.sync()).rejects.toBeInstanceOf(NotImportedError);
  });

  it("file events and polls do nothing until the import has happened, then work", async () => {
    vi.useFakeTimers();
    const s = setup({ imported: false });
    s.service.fileChanged();
    s.service.poll();
    await vi.advanceTimersByTimeAsync(10_000);
    expect(s.gh.jsonCalls).toEqual([]);
    s.h.imported = true;
    s.service.poll();
    await vi.advanceTimersByTimeAsync(10);
    expect(s.gh.jsonCalls.length).toBeGreaterThan(0);
    s.service.dispose();
  });
});

describe("one sync at a time", () => {
  it("never runs two gh operations at once across mutations, syncs and polls", async () => {
    const { gh, req, service } = setup();
    gh.seed({ title: "existing", labels: ["task", "status:todo"] });
    gh.delayMs = 5;
    await Promise.all([
      req("POST", "/kanban/tasks", { title: "a" }),
      req("POST", "/kanban/tasks", { title: "b" }),
      service.sync(), // the sync route's board read afterwards is read-only and not part of this check
      service.sync(),
      req("PATCH", "/kanban/tasks/1", { status: "blocked" }),
      req("POST", "/kanban/tasks", { title: "c" }),
    ]);
    expect(gh.maxInFlight).toBe(1);
    expect(gh.count("create")).toBeGreaterThanOrEqual(3);
  });

  it("triggers that arrive during a run coalesce into one follow-up run", async () => {
    const { gh, service } = setup({ file: "# nothing\n" });
    gh.delayMs = 20;
    const first = service.sync();
    await new Promise((r) => setTimeout(r, 5)); // the first run is now in flight
    const rest = Array.from({ length: 5 }, () => service.sync());
    await Promise.all([first, ...rest]);
    expect(gh.jsonCalls).toHaveLength(2); // the running one + exactly one follow-up
  });
});

describe("file watcher trigger", () => {
  it("debounces bursts into a single sync", async () => {
    vi.useFakeTimers();
    const { gh, service } = setup({ file: "# nothing\n" });
    for (let i = 0; i < 5; i++) {
      service.fileChanged();
      await vi.advanceTimersByTimeAsync(100);
    }
    expect(gh.jsonCalls).toHaveLength(0);
    await vi.advanceTimersByTimeAsync(5000);
    expect(gh.jsonCalls).toHaveLength(1);
    service.dispose();
  });

  it("skips the run when the file equals what the sync itself just wrote; runs for a real edit", async () => {
    vi.useFakeTimers();
    const { gh, h, service } = setup({ file: "## A\n\n- [ ] one\n" });
    await service.sync(); // creates #1 and writes the marker
    expect(h.fileWrites).toBe(1);
    const lists = gh.jsonCalls.length;
    service.fileChanged(); // the watcher noticing our own write
    await vi.advanceTimersByTimeAsync(5000);
    expect(gh.jsonCalls).toHaveLength(lists);
    h.file = h.file + "\n- [ ] user line\n";
    service.fileChanged();
    await vi.advanceTimersByTimeAsync(5000);
    expect(gh.jsonCalls.length).toBe(lists + 1);
    service.dispose();
  });

  it("dispose cancels a pending debounce", async () => {
    vi.useFakeTimers();
    const { gh, service } = setup({ file: "# nothing\n" });
    service.fileChanged();
    service.dispose();
    await vi.advanceTimersByTimeAsync(5000);
    expect(gh.jsonCalls).toHaveLength(0);
  });
});

describe("buildServer", () => {
  const config = { webDist: "/nonexistent", repoRoot: "/tmp" } as Config;
  function server() {
    const s = setup();
    const gh = s.gh;
    const tests = { state: () => ({ status: "idle" }) as never, start: () => true, stop: () => {} };
    const app = buildServer({ config, token: "t".repeat(64), bus: s.bus, gh, tests, kanban: s.service });
    return { app, gh };
  }

  it.each([
    ["GET", "/api/prs"],
    ["GET", "/api/issues"],
    ["POST", "/api/issues"],
    ["GET", "/api/ci"],
    ["GET", "/api/tests"],
    ["POST", "/api/tests/run"],
    ["GET", "/api/events"],
    ["GET", "/api/kanban"],
    ["POST", "/api/kanban/tasks"],
    ["PATCH", "/api/kanban/tasks/1"],
    ["POST", "/api/kanban/sync"],
  ])("%s %s without a token is 401", async (method, url) => {
    const { app, gh } = server();
    const res = await app.request(url, { method });
    expect(res.status).toBe(401);
    expect(gh.writes).toEqual([]);
    expect(gh.jsonCalls).toEqual([]);
  });

  it("with the token, every module answers", async () => {
    const { app } = server();
    const auth = { Authorization: `Bearer ${"t".repeat(64)}` };
    for (const url of ["/api/health", "/api/tests", "/api/kanban", "/api/ci", "/api/issues", "/api/prs"]) {
      const res = await app.request(url, { headers: auth });
      expect(res.status, url).toBe(200);
    }
    const events = await app.request("/api/events", { headers: auth });
    expect(events.status).toBe(200);
    await events.body?.cancel();
  });
});
