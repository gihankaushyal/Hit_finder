import { describe, it, expect, beforeEach, afterEach } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { readJournalFile } from "../server/agents/journal";
import {
  ACTIVE_LEGACY_ID, autoTitle, blankEntry, CONV_ID, journalPath, loadRegistry, newConvId, rebuildRegistry, saveRegistry, statsOf,
  type Registry,
} from "../server/agents/conversations";

let dir: string;
beforeEach(() => {
  dir = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), "dash-convs-")));
});
afterEach(() => fs.rmSync(dir, { recursive: true, force: true }));

/** One journal line, the way `openJournal` writes it. */
const line = (seq: number, body: object) => JSON.stringify({ ...body, seq, at: `2026-10-20T08:0${seq}:00.000Z` }) + "\n";

describe("readJournalFile", () => {
  it("reads entries without touching the file, even when the last line is torn", async () => {
    const file = path.join(dir, "j.jsonl");
    fs.writeFileSync(file, line(1, { kind: "user", text: "hi" }) + '{"kind":"user","te');
    const before = fs.readFileSync(file);
    const events = await readJournalFile(file);
    expect(events).toHaveLength(1);
    expect(events[0]).toMatchObject({ kind: "user", text: "hi", seq: 1 });
    expect(fs.readFileSync(file).equals(before)).toBe(true);
  });

  it("returns nothing for a file that does not exist", async () => {
    expect(await readJournalFile(path.join(dir, "missing.jsonl"))).toEqual([]);
  });
});

describe("ids and titles", () => {
  it("makes ids that pass the path-safety pattern", () => {
    expect(newConvId(Date.UTC(2026, 9, 27), () => 0.5)).toMatch(CONV_ID);
    expect(newConvId(Date.now())).toMatch(CONV_ID);
    expect("../etc".match(CONV_ID)).toBeNull();
    expect("c-A".match(CONV_ID)).toBeNull();
  });

  it("titles a conversation from its first message", () => {
    expect(autoTitle("  plan   the\nweek  ")).toBe("plan the week");
    const long = autoTitle("x".repeat(100));
    expect(long).toHaveLength(61);
    expect(long.endsWith("…")).toBe(true);
  });
});

describe("statsOf", () => {
  it("reads the first message, turns, highest cost and time span", () => {
    const events = [
      { kind: "user", text: "first", seq: 1, at: "2026-10-20T08:01:00.000Z" },
      { kind: "result", ok: true, text: "", turns: 1, costUsd: 0.2, seq: 2, at: "2026-10-20T08:02:00.000Z" },
      { kind: "user", text: "second", seq: 3, at: "2026-10-20T08:03:00.000Z" },
      { kind: "result", ok: true, text: "", turns: 2, costUsd: 0.5, seq: 4, at: "2026-10-20T08:04:00.000Z" },
    ] as Parameters<typeof statsOf>[0];
    expect(statsOf(events)).toEqual({ firstUser: "first", turns: 2, costUsd: 0.5, firstAt: "2026-10-20T08:01:00.000Z", lastAt: "2026-10-20T08:04:00.000Z" });
    expect(statsOf([])).toEqual({ firstUser: null, turns: 0, costUsd: null, firstAt: null, lastAt: null });
  });
});

describe("registry file", () => {
  const reg = (title: string): Registry => ({ v: 1, active: "c-a", convs: { "c-a": { ...blankEntry("c-a", "2026-10-20T08:00:00.000Z"), title } } });

  it("round-trips, and falls back to the previous copy when index.json is corrupt", async () => {
    expect(await loadRegistry(dir)).toBeNull();
    await saveRegistry(dir, reg("one"));
    await saveRegistry(dir, reg("two"));
    expect((await loadRegistry(dir))?.convs["c-a"].title).toBe("two");
    fs.writeFileSync(path.join(dir, "index.json"), "{ not json");
    expect((await loadRegistry(dir))?.convs["c-a"].title).toBe("one");
  });

  it("does not let a corrupt index overwrite the good backup", async () => {
    await saveRegistry(dir, reg("one"));
    await saveRegistry(dir, reg("two"));
    fs.writeFileSync(path.join(dir, "index.json"), "{ not json");
    await saveRegistry(dir, reg("three"));
    expect((await loadRegistry(dir))?.convs["c-a"].title).toBe("three");
    expect(JSON.parse(fs.readFileSync(path.join(dir, "index.json.bak"), "utf8")).convs["c-a"].title).toBe("one");
  });

  it("falls back to the backup when index.json is missing", async () => {
    await saveRegistry(dir, reg("one"));
    await saveRegistry(dir, reg("two"));
    fs.rmSync(path.join(dir, "index.json"));
    expect((await loadRegistry(dir))?.convs["c-a"].title).toBe("one");
  });

  it("rejects an index whose active id is unsafe, and uses the backup", async () => {
    await saveRegistry(dir, reg("one"));
    await saveRegistry(dir, reg("two"));
    fs.writeFileSync(path.join(dir, "index.json"), JSON.stringify({ ...reg("bad"), active: "../../x" }));
    expect((await loadRegistry(dir))?.convs["c-a"].title).toBe("one");
    fs.rmSync(path.join(dir, "index.json.bak"));
    expect(await loadRegistry(dir)).toBeNull();
  });

  it("rejects an index whose active entry was dropped for a bad shape", async () => {
    await saveRegistry(dir, reg("one"));
    await saveRegistry(dir, reg("two"));
    fs.writeFileSync(path.join(dir, "index.json"), JSON.stringify({ v: 1, active: "c-b", convs: { "c-a": reg("x").convs["c-a"], "c-b": { id: "c-b" } } }));
    expect((await loadRegistry(dir))?.convs["c-a"].title).toBe("one");
  });

  it("drops entries with a bad costUsd or readOnly", async () => {
    const good = blankEntry("c-a", "2026-10-20T08:00:00.000Z");
    fs.writeFileSync(
      path.join(dir, "index.json"),
      JSON.stringify({ v: 1, active: "c-a", convs: { "c-a": good, "c-b": { ...good, id: "c-b", costUsd: "1" }, "c-c": { ...good, id: "c-c", readOnly: "nope" } } }),
    );
    expect(Object.keys((await loadRegistry(dir))!.convs)).toEqual(["c-a"]);
  });

  it("does not let an index with a dangling active replace a good backup", async () => {
    await saveRegistry(dir, reg("one"));
    await saveRegistry(dir, reg("two"));
    fs.writeFileSync(path.join(dir, "index.json"), JSON.stringify({ v: 1, active: "c-zz", convs: {} }));
    await saveRegistry(dir, reg("three"));
    expect(JSON.parse(fs.readFileSync(path.join(dir, "index.json.bak"), "utf8")).convs["c-a"].title).toBe("one");
  });

  it("drops entries with unsafe ids or the wrong shape", async () => {
    fs.writeFileSync(
      path.join(dir, "index.json"),
      JSON.stringify({ v: 1, active: "c-a", convs: { "c-a": blankEntry("c-a", "2026-10-20T08:00:00.000Z"), "../x": { id: "../x" }, "c-b": { id: "c-b" } } }),
    );
    expect(Object.keys((await loadRegistry(dir))!.convs)).toEqual(["c-a"]);
  });
});

describe("rebuildRegistry", () => {
  const NOW = "2026-10-27T00:00:00.000Z";
  const writeLegacy = () => {
    fs.writeFileSync(path.join(dir, "journal.jsonl"), line(1, { kind: "user", text: "live topic" }) + line(2, { kind: "result", ok: true, text: "", turns: 1, costUsd: 0.4 }));
    fs.writeFileSync(path.join(dir, "meta.json"), JSON.stringify({ turns: 3, costUsd: 1.5, sessionId: "sess-live" }));
    fs.writeFileSync(path.join(dir, "journal-2026-10-20T08-01-00-000Z.jsonl"), line(1, { kind: "user", text: "old topic" }) + line(2, { kind: "result", ok: true, text: "", turns: 1, costUsd: 0.2 }));
  };

  it("starts empty with one blank active conversation", async () => {
    const r = await rebuildRegistry(dir, NOW);
    expect(Object.keys(r.convs)).toEqual([r.active]);
    expect(r.convs[r.active]).toMatchObject({ title: null, turns: 0, sessionId: null, archived: false, readOnly: null });
  });

  it("imports the live conversation with its session id, and old logs as read-only archives", async () => {
    writeLegacy();
    const r = await rebuildRegistry(dir, NOW);
    expect(r.active).toBe(ACTIVE_LEGACY_ID);
    expect(r.convs[ACTIVE_LEGACY_ID]).toMatchObject({ title: "live topic", turns: 3, costUsd: 1.5, sessionId: "sess-live", archived: false, readOnly: null });
    expect(r.convs["c-2026-10-20t08-01-00-000z"]).toMatchObject({ title: "old topic", turns: 1, costUsd: 0.2, sessionId: null, archived: true, readOnly: "legacy" });
    expect(fs.existsSync(path.join(dir, "journal.jsonl"))).toBe(false);
    expect(fs.existsSync(path.join(dir, "meta.json"))).toBe(false);
    expect(fs.existsSync(journalPath(dir, ACTIVE_LEGACY_ID))).toBe(true);
    expect(await loadRegistry(dir)).toEqual(r);
  });

  it("survives a run that was interrupted after the journal had moved", async () => {
    writeLegacy();
    fs.mkdirSync(path.dirname(journalPath(dir, ACTIVE_LEGACY_ID)), { recursive: true });
    fs.renameSync(path.join(dir, "journal.jsonl"), journalPath(dir, ACTIVE_LEGACY_ID));
    const r = await rebuildRegistry(dir, NOW);
    expect(r.convs[ACTIVE_LEGACY_ID]).toMatchObject({ sessionId: "sess-live", title: "live topic" });
    expect(Object.keys(r.convs)).toHaveLength(2);
  });

  it("recovers conversations from their folders when no index survives", async () => {
    await saveRegistry(dir, { v: 1, active: "c-a", convs: { "c-a": blankEntry("c-a", NOW) } });
    fs.mkdirSync(path.dirname(journalPath(dir, "c-a")), { recursive: true });
    fs.writeFileSync(journalPath(dir, "c-a"), line(1, { kind: "user", text: "kept" }));
    fs.rmSync(path.join(dir, "index.json"));
    fs.rmSync(path.join(dir, "index.json.bak"), { force: true });
    const r = await rebuildRegistry(dir, NOW);
    expect(r.convs["c-a"]).toMatchObject({ title: "kept", sessionId: null, archived: true, readOnly: "legacy" });
  });
});
