import { describe, it, expect, vi, afterEach } from "vitest";
import crypto from "node:crypto";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { FakeGh } from "./helpers/fakeGh";
import { EventBus } from "../server/events";
import { main } from "../server/cli/kanbanSync";
import { createKanbanService, NotImportedError } from "../server/lib/kanbanService";
import {
  runSync, fileSyncDeps, isImportCompleted, escapeMentions, PENDING_MAX_AGE_MS, SyncLockedError, SyncStateError,
  type SyncDeps, type SyncState,
} from "../server/lib/kanbanSync";
import type { Config } from "../server/config";

const sha = (s: string): string => crypto.createHash("sha256").update(s).digest("hex");
const THREE = [
  "## Open decisions (need a call)",
  "",
  "- [x] **Decided thing** — done already",
  "- [ ] **Open thing**",
  "",
  "## In progress (Phase 5)",
  "",
  "- [ ] Working on it",
  "",
].join("\n");
const MARKER_RE = / <!-- gh:#\d+ -->/g;
const strip = (t: string): string => t.replace(MARKER_RE, "");

function mk(text: string | null, gh = new FakeGh()) {
  const h = {
    gh,
    file: text as string | null,
    state: { labelsEnsured: false, items: {} } as SyncState,
    fileWrites: 0,
    stateSaves: 0,
    now: 1_000_000,
    /** Called before every state save; throw to simulate the process dying there. */
    beforeSave: (_s: SyncState) => {},
  };
  const deps: SyncDeps = {
    gh,
    readFile: () => h.file,
    writeFile: (t) => {
      h.file = t;
      h.fileWrites++;
    },
    loadState: () => structuredClone(h.state),
    saveState: (s) => {
      h.beforeSave(s);
      h.state = structuredClone(s);
      h.stateSaves++;
    },
    sleep: async () => {},
    now: () => h.now,
  };
  return { h, deps, gh };
}
const sync = (deps: SyncDeps, extra: { dryRun?: boolean; rebuildState?: boolean } = {}) => runSync(deps, { dryRun: false, ...extra });
const quiet = () => vi.spyOn(console, "error").mockImplementation(() => {});

/** The invariant every adoption scenario ends with. */
function expectNoHarm(h: { file: string | null }, gh: FakeGh, expected: { titles: string[]; ticked: string[] }) {
  const titles = [...gh.issues.values()].map((i) => i.title).sort();
  expect(titles).toEqual([...expected.titles].sort()); // no duplicate issue exists
  expect(new Set(titles).size).toBe(titles.length);
  expect(h.file).not.toContain("## Inbox"); // no extra Inbox item was appended
  for (const t of expected.ticked) expect(h.file).toMatch(new RegExp(`- \\[x\\] (\\*\\*)?${t}`)); // every ticked box is still ticked
}

describe("1. stateless adoption", () => {
  afterEach(() => vi.restoreAllMocks());

  it("process killed between gh create returning and anything being saved", async () => {
    const { h, deps, gh } = mk(THREE);
    h.beforeSave = (s) => {
      if (Object.keys(s.pending ?? {}).length > 0) throw new Error("killed");
    };
    await expect(sync(deps)).rejects.toThrow("killed");
    expect(gh.issues.size).toBe(1);
    expect(h.file).toBe(THREE); // no marker, no pending record anywhere
    expect(h.state.pending ?? {}).toEqual({});
    h.beforeSave = () => {};
    await sync(deps);
    expectNoHarm(h, gh, { titles: ["Decided thing", "Open thing", "Working on it"], ticked: ["Decided thing"] });
    expect(gh.count("create")).toBe(3);
    expect(gh.issues.get(1)!.state).toBe("CLOSED"); // the ticked item's adopted issue was closed
    expect(h.file).toContain("<!-- gh:#1 -->");
    expect((await sync(deps)).actions).toEqual([]);
  });

  it("file edited during the create call, and no pending record survives", async () => {
    const { h, deps, gh } = mk(THREE);
    const edited = THREE + "\n- [ ] user added this meanwhile\n";
    gh.onCreate = (n) => {
      if (n === 1) h.file = edited;
    };
    const r1 = await sync(deps);
    expect(r1.aborted).toBe(true);
    gh.onCreate = () => {};
    h.state.pending = {}; // stateless: the pending record is gone
    await sync(deps);
    expectNoHarm(h, gh, { titles: ["Decided thing", "Open thing", "Working on it", "user added this meanwhile"], ticked: ["Decided thing"] });
    expect(gh.count("create")).toBe(4);
    expect(gh.issues.get(1)!.state).toBe("CLOSED");
  });

  it("markers stripped from every item", async () => {
    const { h, deps, gh } = mk(THREE);
    await sync(deps);
    const creates = gh.count("create");
    h.file = strip(h.file!);
    expect(h.file).not.toContain("gh:#");
    const r = await sync(deps);
    expectNoHarm(h, gh, { titles: ["Decided thing", "Open thing", "Working on it"], ticked: ["Decided thing"] });
    expect(gh.count("create")).toBe(creates);
    expect(r.actions.map((a) => a.type)).toEqual(["link", "link", "link"]);
    expect(h.file).toContain("<!-- gh:#1 -->");
    expect(h.file).toContain("<!-- gh:#3 -->");
    expect(r.notInFile).toEqual([]);
    expect((await sync(deps)).actions).toEqual([]);
  });

  it("markers stripped while a box was ticked: the adopted open issue is closed, nothing is unticked", async () => {
    const { h, deps, gh } = mk(THREE);
    await sync(deps);
    h.file = strip(h.file!).replace("- [ ] **Open thing**", "- [x] **Open thing**");
    await sync(deps);
    expectNoHarm(h, gh, { titles: ["Decided thing", "Open thing", "Working on it"], ticked: ["Decided thing", "Open thing"] });
    expect(gh.issues.get(2)!.state).toBe("CLOSED");
    expect(gh.count("reopen")).toBe(0);
  });

  it("a closed adopted issue ticks an unticked box and never unticks a ticked one", async () => {
    const { h, deps, gh } = mk("## A\n\n- [ ] alpha\n");
    await sync(deps);
    gh.issues.get(1)!.state = "CLOSED";
    h.file = strip(h.file!);
    await sync(deps);
    expect(h.file).toContain("- [x] alpha <!-- gh:#1 -->");
    expect(gh.count("create")).toBe(1);
  });

  it("two items with identical title and body adopt two different issues", async () => {
    const { h, deps, gh } = mk("## A\n\n- [ ] same\n- [ ] same\n");
    await sync(deps);
    expect(gh.issues.size).toBe(2);
    h.file = strip(h.file!);
    await sync(deps);
    expect(gh.count("create")).toBe(2);
    expect(h.file).toBe("## A\n\n- [ ] same <!-- gh:#1 -->\n- [ ] same <!-- gh:#2 -->\n");
  });

  it("an issue that differs in title or body is not adopted", async () => {
    const { h, deps, gh } = mk("## A\n\n- [ ] alpha\n");
    gh.seed({ title: "alpha", body: "something else" });
    const spy = quiet();
    await sync(deps);
    spy.mockRestore();
    expect(gh.count("create")).toBe(1);
    expect(h.file).toContain("<!-- gh:#2 -->");
  });
});

describe("2. a pending issue that never appears gets an exit", () => {
  const ONE = "## A\n\n- [ ] one\n";
  async function pendingInvisible() {
    const m = mk(ONE);
    m.gh.onCreate = () => {
      m.h.file = ONE + "\n";
    };
    await sync(m.deps);
    m.gh.onCreate = () => {};
    m.gh.issues.delete(1); // gh list lags: the issue is not visible
    m.h.file = ONE;
    return m;
  }

  it("is skipped while young, then reported as a conflict naming the issue and not created", async () => {
    const { h, deps, gh } = await pendingInvisible();
    expect(h.state.pending!["1"].at).toBe(h.now);
    h.now += PENDING_MAX_AGE_MS - 1;
    const young = await sync(deps);
    expect(young.skipped).toHaveLength(1);
    expect(young.conflicts).toEqual([]);
    h.now += 1;
    const old = await sync(deps);
    expect(old.skipped).toEqual([]);
    expect(old.conflicts.join()).toMatch(/#1 .*not appeared/);
    expect(old.actions).toEqual([]);
    expect(gh.count("create")).toBe(1); // nothing created automatically
  });

  it("a fresh pending issue never blocks forever: it links as soon as it appears", async () => {
    const { h, deps, gh } = await pendingInvisible();
    gh.seed({ number: 1, title: "one", body: "one\n\n_Synced from phase-05-kanban.md, section: A_" });
    await sync(deps);
    expect(h.file).toContain("<!-- gh:#1 -->");
  });
});

describe("3. cross-process lock", () => {
  let dir: string;
  const setup = (kanban = "## A\n\n- [ ] one\n") => {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), "dash-lock-"));
    const kanbanPath = path.join(dir, "k.md");
    const stateDir = path.join(dir, "state");
    fs.mkdirSync(stateDir);
    fs.writeFileSync(kanbanPath, kanban);
    const gh = new FakeGh();
    const config = { kanbanPath, stateDir } as Config;
    return { gh, config, deps: fileSyncDeps(config, gh), lock: path.join(stateDir, "kanban-sync.lock"), kanbanPath };
  };
  afterEach(() => {
    vi.restoreAllMocks();
    fs.rmSync(dir, { recursive: true, force: true });
  });

  it("a lock held by a live pid stops the run before any gh call, and is left alone", async () => {
    const { gh, deps, lock } = setup();
    const payload = JSON.stringify({ pid: process.pid, startedAt: "2026-10-08T00:00:00Z" });
    fs.writeFileSync(lock, payload);
    await expect(runSync(deps, { dryRun: false })).rejects.toThrow(SyncLockedError);
    await expect(runSync(deps, { dryRun: false })).rejects.toThrow(/another kanban sync is in progress/);
    expect(gh.jsonCalls).toEqual([]);
    expect(gh.writes).toEqual([]);
    expect(fs.readFileSync(lock, "utf8")).toBe(payload);
  });

  it("a lock held by a dead pid is taken over and released afterwards", async () => {
    const { gh, deps, lock } = setup();
    const dead = spawnSync(process.execPath, ["-e", ""]).pid!;
    fs.writeFileSync(lock, JSON.stringify({ pid: dead, startedAt: "2026-10-08T00:00:00Z" }));
    await runSync(deps, { dryRun: false });
    expect(gh.count("create")).toBe(1);
    expect(fs.existsSync(lock)).toBe(false);
  });

  it("the lock file records pid and start time while the run holds it", async () => {
    const { gh, deps, lock } = setup();
    let seen: { pid: number; startedAt: string } | null = null;
    const orig = gh.json.bind(gh);
    gh.json = async (a) => {
      seen = JSON.parse(fs.readFileSync(lock, "utf8"));
      return orig(a);
    };
    await runSync(deps, { dryRun: false });
    expect(seen!.pid).toBe(process.pid);
    expect(Date.parse(seen!.startedAt)).not.toBeNaN();
  });

  it("is released when the run throws", async () => {
    const { gh, deps, lock } = setup();
    gh.failNextJson = true;
    await expect(runSync(deps, { dryRun: false })).rejects.toThrow("gh list failed");
    expect(fs.existsSync(lock)).toBe(false);
  });

  it("a dry run takes no lock and ignores a held one", async () => {
    const { deps, lock } = setup();
    fs.writeFileSync(lock, JSON.stringify({ pid: process.pid, startedAt: "x" }));
    await expect(runSync(deps, { dryRun: true })).resolves.toBeDefined();
  });

  it("the CLI exits non-zero with the in-progress message", async () => {
    const { deps, lock } = setup();
    fs.writeFileSync(lock, JSON.stringify({ pid: process.pid, startedAt: "x" }));
    const err: string[] = [];
    const code = await main(["--yes"], { sync: deps, importCompleted: () => true, out: () => {}, err: (l) => err.push(l) });
    expect(code).toBe(1);
    expect(err.join("\n")).toMatch(/another kanban sync is in progress/);
  });

  it("the service treats a held lock as a skipped run, not an error", async () => {
    const m = mk("## A\n\n- [ ] one\n");
    m.deps.acquireLock = () => {
      throw new SyncLockedError("another kanban sync is in progress");
    };
    const spy = quiet();
    const service = createKanbanService({ gh: m.gh, sync: m.deps, bus: new EventBus(), imported: () => true });
    await service.sync();
    expect((await service.board()).syncError).toBeNull();
    expect(spy).not.toHaveBeenCalled();
    expect(m.gh.writes).toEqual([]);
  });
});

describe("4. state file handling", () => {
  let dir: string;
  const setup = (kanban: string) => {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), "dash-state-"));
    const kanbanPath = path.join(dir, "k.md");
    const stateDir = path.join(dir, "state");
    fs.mkdirSync(stateDir);
    fs.writeFileSync(kanbanPath, kanban);
    const gh = new FakeGh();
    const config = { kanbanPath, stateDir } as Config;
    return { gh, config, deps: fileSyncDeps(config, gh), statePath: path.join(stateDir, "kanban-sync.json"), kanbanPath };
  };
  afterEach(() => fs.rmSync(dir, { recursive: true, force: true }));

  it("a missing state file is a fresh state", () => {
    const { deps } = setup("x");
    expect(deps.loadState()).toEqual({ labelsEnsured: false, items: {} });
  });

  it.each([
    ["not json", "{oops"],
    ["empty", ""],
    ["an array", "[]"],
    ["items missing", '{"labelsEnsured":true}'],
    ["items of the wrong type", '{"items":5}'],
    ["a bad item record", '{"items":{"1":{"checked":"yes"}}}'],
    ["a bad pending entry", '{"items":{},"pending":{"1":{"hash":5}}}'],
  ])("%s: throws a clear error and leaves the file untouched, with no gh call", async (_n, content) => {
    const { gh, deps, statePath, kanbanPath } = setup("## A\n\n- [ ] one\n");
    fs.writeFileSync(statePath, content);
    await expect(runSync(deps, { dryRun: false })).rejects.toThrow(SyncStateError);
    await expect(runSync(deps, { dryRun: false })).rejects.toThrow(/sync state file .* nothing was changed/);
    expect(fs.readFileSync(statePath, "utf8")).toBe(content);
    expect(fs.readFileSync(kanbanPath, "utf8")).toBe("## A\n\n- [ ] one\n");
    expect(gh.writes).toEqual([]);
    expect(isImportCompleted({ stateDir: path.dirname(statePath) })).toBe(false);
  });

  it("markers in the file but no records in the state: refuses to run", async () => {
    const { h, deps, gh } = mk("## A\n\n- [x] one <!-- gh:#1 -->\n");
    gh.seed({ title: "one" });
    await expect(sync(deps)).rejects.toThrow(/--rebuild-state/);
    await expect(sync(deps, { dryRun: true })).rejects.toThrow(SyncStateError);
    expect(gh.writes).toEqual([]);
    expect(h.fileWrites).toBe(0);
    expect(h.stateSaves).toBe(0);
  });

  it("--rebuild-state records reality and flips, closes and reopens nothing", async () => {
    const file = "## A\n\n- [x] ticked, open <!-- gh:#1 -->\n- [ ] unticked, closed <!-- gh:#2 -->\n- [ ] fine <!-- gh:#3 -->\n";
    const { h, deps, gh } = mk(file);
    gh.seed({ title: "ticked, open" });
    gh.seed({ title: "unticked, closed", state: "CLOSED" });
    gh.seed({ title: "fine" });
    const r = await sync(deps, { rebuildState: true });
    expect(h.file).toBe(file);
    expect(gh.count("close")).toBe(0);
    expect(gh.count("reopen")).toBe(0);
    expect(gh.count("edit")).toBe(0);
    expect(r.actions).toEqual([]);
    expect(Object.keys(h.state.items)).toEqual(["1", "2", "3"]);
    expect(h.state.items["1"]).toMatchObject({ checked: true, closed: false });
    expect(h.state.items["2"]).toMatchObject({ checked: false, closed: true });
    expect(h.state.importCompletedAt).toBeDefined();
    // The disagreements are left for the user, not resolved by a later plain run either.
    const again = await sync(deps);
    expect(again.actions).toEqual([]);
    expect(again.conflicts).toHaveLength(2);
    expect(h.file).toBe(file);
    expect(gh.count("close")).toBe(0);
  });

  it("pending records count as state, so a crash right after the marker was written still resumes", async () => {
    const { h, deps, gh } = mk("## A\n\n- [ ] one <!-- gh:#1 -->\n");
    gh.seed({ title: "one" });
    h.state.pending = { "1": { hash: "whatever", at: h.now } };
    await expect(sync(deps)).resolves.toBeDefined();
  });
});

describe("5. safe writes", () => {
  let dir: string;
  const setup = (kanban: string | Buffer, mode = 0o664) => {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), "dash-write-"));
    const kanbanPath = path.join(dir, "k.md");
    const stateDir = path.join(dir, "state");
    fs.writeFileSync(kanbanPath, kanban);
    fs.chmodSync(kanbanPath, mode);
    const gh = new FakeGh();
    const config = { kanbanPath, stateDir } as Config;
    return { gh, config, deps: fileSyncDeps(config, gh), kanbanPath, stateDir };
  };
  afterEach(() => {
    vi.restoreAllMocks();
    fs.rmSync(dir, { recursive: true, force: true });
  });
  const leftovers = (d: string): string[] => fs.readdirSync(d).filter((f) => f.endsWith(".tmp"));

  it("keeps the original mode (0o664) regardless of umask", async () => {
    const { deps, kanbanPath } = setup("## A\n\n- [ ] one\n");
    const old = process.umask(0o077);
    try {
      await runSync(deps, { dryRun: false });
    } finally {
      process.umask(old);
    }
    expect(fs.statSync(kanbanPath).mode & 0o777).toBe(0o664);
    expect(fs.readFileSync(kanbanPath, "utf8")).toContain("<!-- gh:#1 -->");
  });

  it("fsyncs the temp file before the rename", () => {
    const { deps } = setup("x\n");
    const order: string[] = [];
    const fsync = fs.fsyncSync;
    const rename = fs.renameSync;
    vi.spyOn(fs, "fsyncSync").mockImplementation((fd) => {
      order.push("fsync");
      return fsync(fd);
    });
    vi.spyOn(fs, "renameSync").mockImplementation((a, b) => {
      order.push("rename");
      return rename(a, b);
    });
    deps.writeFile("y\n");
    expect(order.indexOf("fsync")).toBeGreaterThanOrEqual(0);
    expect(order.indexOf("fsync")).toBeLessThan(order.indexOf("rename"));
  });

  it("removes the temp file and leaves the original intact when the write fails", () => {
    const { deps, kanbanPath } = setup("original\n");
    vi.spyOn(fs, "renameSync").mockImplementation(() => {
      throw new Error("disk full");
    });
    expect(() => deps.writeFile("new\n")).toThrow("disk full");
    expect(fs.readFileSync(kanbanPath, "utf8")).toBe("original\n");
    expect(leftovers(path.dirname(kanbanPath))).toEqual([]);
  });

  it("the state file is written the same way (no temp left behind on failure)", () => {
    const { deps, stateDir } = setup("x\n");
    vi.spyOn(fs, "renameSync").mockImplementation(() => {
      throw new Error("disk full");
    });
    expect(() => deps.saveState({ labelsEnsured: true, items: {} })).toThrow("disk full");
    expect(leftovers(stateDir)).toEqual([]);
  });

  it("invalid UTF-8 aborts the run with a clear message and writes nothing", async () => {
    const bytes = Buffer.from([0x23, 0x20, 0xff, 0xfe, 0x0a, 0x2d, 0x20, 0x5b, 0x20, 0x5d, 0x20, 0x61, 0x0a]);
    const { gh, deps, kanbanPath } = setup(bytes);
    await expect(runSync(deps, { dryRun: false })).rejects.toThrow(/not valid UTF-8/);
    expect(gh.writes).toEqual([]);
    expect(fs.readFileSync(kanbanPath).equals(bytes)).toBe(true);
  });

  it("a UTF-8 byte order mark survives a rewrite", async () => {
    const { deps, kanbanPath } = setup("﻿## A\n\n- [ ] one\n");
    await runSync(deps, { dryRun: false });
    expect(fs.readFileSync(kanbanPath, "utf8").startsWith("﻿## A")).toBe(true);
  });
});

describe("6. one-time backup", () => {
  let dir: string;
  const setup = (kanban: string) => {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), "dash-bak-"));
    const kanbanPath = path.join(dir, "k.md");
    fs.writeFileSync(kanbanPath, kanban);
    fs.chmodSync(kanbanPath, 0o640);
    const gh = new FakeGh();
    return { gh, deps: fileSyncDeps({ kanbanPath, stateDir: path.join(dir, "state") } as Config, gh), kanbanPath, bak: kanbanPath + ".pre-sync.bak" };
  };
  afterEach(() => fs.rmSync(dir, { recursive: true, force: true }));
  const ORIGINAL = "## A\n\n- [ ] one\n";

  it("copies the file before the first write, same mode, and never overwrites it afterwards", async () => {
    const { deps, kanbanPath, bak } = setup(ORIGINAL);
    await runSync(deps, { dryRun: false });
    expect(fs.readFileSync(bak, "utf8")).toBe(ORIGINAL);
    expect(fs.statSync(bak).mode & 0o777).toBe(0o640);
    fs.writeFileSync(kanbanPath, fs.readFileSync(kanbanPath, "utf8") + "- [ ] two\n");
    await runSync(deps, { dryRun: false });
    expect(fs.readFileSync(kanbanPath, "utf8")).toContain("two <!-- gh:#2 -->");
    expect(fs.readFileSync(bak, "utf8")).toBe(ORIGINAL);
    expect(fs.readdirSync(dir).filter((f) => f.endsWith(".tmp"))).toEqual([]);
  });

  it("an existing backup is left alone", async () => {
    const { deps, bak } = setup(ORIGINAL);
    fs.writeFileSync(bak, "older backup");
    await runSync(deps, { dryRun: false });
    expect(fs.readFileSync(bak, "utf8")).toBe("older backup");
  });

  it("a dry run makes no backup", async () => {
    const { deps, bak } = setup(ORIGINAL);
    await runSync(deps, { dryRun: true });
    expect(fs.existsSync(bak)).toBe(false);
  });

  it("no backup when the run never writes the file", async () => {
    const { deps, bak, gh } = setup("# nothing to sync\n");
    await runSync(deps, { dryRun: false });
    expect(gh.count("create")).toBe(0);
    expect(fs.existsSync(bak)).toBe(false);
  });
});

describe("7. import-complete gate", () => {
  let dir: string;
  afterEach(() => {
    vi.restoreAllMocks();
    fs.rmSync(dir, { recursive: true, force: true });
  });
  function setup(kanban: string) {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), "dash-gate-"));
    const kanbanPath = path.join(dir, "k.md");
    const stateDir = path.join(dir, "state");
    fs.writeFileSync(kanbanPath, kanban);
    const config = { kanbanPath, stateDir } as Config;
    return { config, kanbanPath };
  }
  class AuthFailsOnSecondCreate extends FakeGh {
    armed = true;
    async text(args: string[]): Promise<string> {
      if (this.armed && args[1] === "create" && this.creates === 1) throw new Error("gh auth login required");
      return super.text(args);
    }
  }
  const THREE_ITEMS = "## A\n\n- [ ] one\n- [ ] two\n- [ ] three\n";

  it("state is written at the start but importCompletedAt only when the import finishes", async () => {
    const { config } = setup(THREE_ITEMS);
    const gh = new AuthFailsOnSecondCreate();
    const spy = quiet();
    await expect(runSync(fileSyncDeps(config, gh), { dryRun: false })).rejects.toThrow(/auth/);
    spy.mockRestore();
    expect(fs.existsSync(path.join(config.stateDir, "kanban-sync.json"))).toBe(true);
    expect(isImportCompleted(config)).toBe(false);
  });

  it("an interrupted first import needs --yes again and the service does not resume it", async () => {
    const { config } = setup(THREE_ITEMS);
    const spy = quiet();
    const gh = new AuthFailsOnSecondCreate();
    await expect(runSync(fileSyncDeps(config, gh), { dryRun: false })).rejects.toThrow();
    spy.mockRestore();
    gh.armed = false;
    gh.writes = [];
    const deps = fileSyncDeps(config, gh);
    const err: string[] = [];
    const io = { sync: deps, importCompleted: () => isImportCompleted(config), out: () => {}, err: (l: string) => err.push(l) };
    expect(await main([], io)).toBe(1);
    expect(err.join("\n")).toMatch(/--yes/);
    expect(gh.writes).toEqual([]);
    const service = createKanbanService({ gh, sync: deps, bus: new EventBus(), imported: () => isImportCompleted(config), debounceMs: 1 });
    await expect(service.sync()).rejects.toThrow(NotImportedError);
    service.poll();
    service.fileChanged();
    await new Promise((r) => setTimeout(r, 20));
    expect(gh.writes).toEqual([]);
    service.dispose();
    // With --yes the import resumes and completes.
    expect(await main(["--yes"], io)).toBe(0);
    expect(isImportCompleted(config)).toBe(true);
    expect(gh.issues.size).toBe(3);
  });

  it("a complete first run sets importCompletedAt once; later runs do not touch it", async () => {
    const { config } = setup(THREE_ITEMS);
    await runSync(fileSyncDeps(config, new FakeGh()), { dryRun: false });
    expect(isImportCompleted(config)).toBe(true);
    const statePath = path.join(config.stateDir, "kanban-sync.json");
    const before = fs.readFileSync(statePath, "utf8");
    await runSync(fileSyncDeps(config, new FakeGh()), { dryRun: false });
    expect(fs.readFileSync(statePath, "utf8")).toBe(before);
  });

  it("existence of the state file alone does not count", () => {
    const { config } = setup("x");
    fs.mkdirSync(config.stateDir, { recursive: true });
    fs.writeFileSync(path.join(config.stateDir, "kanban-sync.json"), '{"labelsEnsured":true,"items":{}}');
    expect(isImportCompleted(config)).toBe(false);
  });

  it("an aborted run does not complete the import", async () => {
    const m = mk("## A\n\n- [ ] one\n- [ ] two\n");
    m.gh.onCreate = () => {
      m.h.file = m.h.file + "\n";
    };
    const r = await sync(m.deps);
    expect(r.aborted).toBe(true);
    expect(m.h.state.importCompletedAt).toBeUndefined();
  });
});

describe("8. per-item isolation", () => {
  afterEach(() => vi.restoreAllMocks());
  const FILE = "## A\n\n- [x] one <!-- gh:#1 -->\n- [x] two <!-- gh:#2 -->\n- [x] three <!-- gh:#3 -->\n";
  function linked(gh: FakeGh) {
    const m = mk(FILE, gh);
    for (const t of ["one", "two", "three"]) gh.seed({ title: t });
    for (const n of ["1", "2", "3"]) m.h.state.items[n] = { checked: false, closed: false, bodyHash: sha(["one", "two", "three"][Number(n) - 1]) };
    return m;
  }

  it("a failing close on one item is a conflict for that item and the run continues", async () => {
    class FailsOnTwo extends FakeGh {
      broken = true;
      async text(args: string[]): Promise<string> {
        if (this.broken && args[1] === "close" && args[2] === "2") throw new Error("gh: something odd");
        return super.text(args);
      }
    }
    const gh = new FailsOnTwo();
    const { h, deps } = linked(gh);
    const spy = quiet();
    const r = await sync(deps);
    spy.mockRestore();
    expect(r.aborted).toBe(false);
    expect(r.conflicts.join()).toMatch(/#2/);
    expect(r.conflicts.join()).not.toMatch(/something odd/); // no raw error text in reports
    expect(gh.issues.get(1)!.state).toBe("CLOSED");
    expect(gh.issues.get(3)!.state).toBe("CLOSED");
    expect(gh.issues.get(2)!.state).toBe("OPEN");
    expect(h.state.items["1"]).toMatchObject({ checked: true, closed: true });
    expect(h.state.items["3"]).toMatchObject({ checked: true, closed: true });
    expect(h.state.items["2"]).toMatchObject({ checked: false, closed: false }); // still detectable
    gh.broken = false;
    const next = await sync(deps);
    expect(next.actions).toEqual([{ type: "close", issue: 2 }]);
    expect(gh.issues.get(2)!.state).toBe("CLOSED");
  });

  it("a failing create on one item does not stop the others", async () => {
    const { h, deps, gh } = mk("## A\n\n- [ ] one\n- [ ] two\n- [ ] three\n");
    gh.failCreateAt = 2;
    const spy = quiet();
    const r = await sync(deps);
    spy.mockRestore();
    expect(r.conflicts.join()).toMatch(/creating "two" failed/);
    expect(h.file).toContain("one <!-- gh:#1 -->");
    expect(h.file).toContain("three <!-- gh:#2 -->"); // the failed create took no number
    expect(h.file).not.toContain("two <!--");
  });

  it.each(["gh auth login required", "HTTP 403: API rate limit exceeded", "dial tcp: lookup api.github.com: network is unreachable"])(
    "a global failure (%s) still aborts the run",
    async (message) => {
      class Global extends FakeGh {
        async text(args: string[]): Promise<string> {
          if (args[1] === "close" && args[2] === "2") throw new Error(message);
          return super.text(args);
        }
      }
      const gh = new Global();
      const { deps } = linked(gh);
      const spy = quiet();
      await expect(sync(deps)).rejects.toThrow(message);
      spy.mockRestore();
      expect(gh.issues.get(3)!.state).toBe("OPEN"); // never reached
    },
  );

  it("a failing list call aborts before anything is written", async () => {
    const { h, deps, gh } = mk("## A\n\n- [ ] one\n");
    gh.failNextJson = true;
    await expect(sync(deps)).rejects.toThrow("gh list failed");
    expect(gh.writes).toEqual([]);
    expect(h.fileWrites).toBe(0);
  });
});

describe("9. @ mentions are neutralised", () => {
  const FILE = "## A\n\n- [ ] use @property here\n  body mentions @someone and a@b.com\n";

  it("escapeMentions is idempotent", () => {
    const once = escapeMentions("@a @b");
    expect(escapeMentions(once)).toBe(once);
    expect(once).not.toMatch(/@[a-z]/);
  });

  it("titles and bodies sent to gh carry no live mention, and a second sync pushes no update", async () => {
    const { h, deps, gh } = mk(FILE);
    await sync(deps);
    const issue = gh.issues.get(1)!;
    expect(issue.title).not.toMatch(/@[A-Za-z]/);
    expect(issue.body).not.toMatch(/@[A-Za-z]/);
    expect(issue.title).toContain("@‍property");
    expect(h.file).toContain("use @property here"); // the file keeps the real text
    const before = gh.writes.length;
    const r = await sync(deps);
    expect(r.actions).toEqual([]);
    expect(gh.writes).toHaveLength(before);
    expect(gh.count("edit")).toBe(0);
  });

  it("adoption compares with the same transformation", async () => {
    const { h, deps, gh } = mk(FILE);
    await sync(deps);
    h.file = strip(h.file!);
    const r = await sync(deps);
    expect(r.actions.map((a) => a.type)).toEqual(["link"]);
    expect(gh.count("create")).toBe(1);
  });

  it("a GitHub-side title is appended to the file without the joiner", async () => {
    const { h, deps, gh } = mk(FILE);
    await sync(deps);
    gh.seed({ title: "from @‍github" });
    await sync(deps);
    expect(h.file).toContain("- [ ] from @github <!-- gh:#2 -->");
  });
});

describe("10. body update path updates the hash even when checkbox and issue disagree", () => {
  it("one unresolved conflict does not cause the same body to be pushed on every sync", async () => {
    const { h, deps, gh } = mk("## A\n\n- [x] edited body <!-- gh:#1 -->\n");
    gh.seed({ title: "edited body" }); // open, while the box is ticked
    h.state.items["1"] = { checked: true, closed: false, bodyHash: sha("old body") };
    const r1 = await sync(deps);
    expect(r1.actions).toEqual([{ type: "update-body", issue: 1 }]);
    expect(r1.conflicts.join()).toMatch(/#1 differs/);
    expect(gh.count("edit")).toBe(1);
    expect(h.state.items["1"]).toMatchObject({ checked: true, closed: false, bodyHash: sha("edited body") });
    const r2 = await sync(deps);
    expect(r2.actions).toEqual([]);
    expect(gh.count("edit")).toBe(1);
  });
});
