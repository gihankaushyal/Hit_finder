import { describe, it, expect, beforeEach, afterEach } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { readJournalFile } from "../server/agents/journal";
import {
  ACTIVE_LEGACY_ID, adoptEntries, autoTitle, buildDigest, selectSweep, sweepTempFiles, blankEntry, cleanTitle, isEmptyConv, pickContinuable, CONV_ID, journalPath, loadRegistry, newConvId, rebuildRegistry, saveRegistry, statsOf,
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

  it("cuts a title by code point and gives no title for text with nothing printable", () => {
    const emoji = autoTitle("😀".repeat(70));
    expect([...emoji]).toHaveLength(61);
    expect(emoji).not.toMatch(/[\ud800-\udbff](?![\udc00-\udfff])/); // no lone surrogate
    expect(autoTitle("\u0001\u0002 \n")).toBe("");
  });

  it("removes temp files a killed process left, but not this process's own", async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "dash-sweep-"));
    fs.mkdirSync(path.join(dir, "conv", "c-a"), { recursive: true });
    fs.writeFileSync(path.join(dir, "index.json.99999.tmp"), "x");
    fs.writeFileSync(path.join(dir, "conv", "c-a", "entry.json.99999.tmp"), "x");
    fs.writeFileSync(path.join(dir, "conv", "c-a", `entry.json.${process.pid}.tmp`), "x");
    fs.writeFileSync(path.join(dir, "index.json"), "{}");
    await sweepTempFiles(dir);
    expect(fs.readdirSync(dir).sort()).toEqual(["conv", "index.json"]);
    expect(fs.readdirSync(path.join(dir, "conv", "c-a"))).toEqual([`entry.json.${process.pid}.tmp`]);
    fs.rmSync(dir, { recursive: true, force: true });
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

  it("keeps the conversations of an index whose active id is unsafe, with no active one", async () => {
    await saveRegistry(dir, reg("one"));
    await saveRegistry(dir, reg("two"));
    fs.writeFileSync(path.join(dir, "index.json"), JSON.stringify({ ...reg("bad"), active: "../../x" }));
    const r = (await loadRegistry(dir))!;
    expect(r.active).toBe("");
    expect(r.convs["c-a"].title).toBe("bad");
  });

  it("keeps the conversations of an index whose active entry was dropped for a bad shape", async () => {
    fs.writeFileSync(path.join(dir, "index.json"), JSON.stringify({ v: 1, active: "c-b", convs: { "c-a": reg("x").convs["c-a"], "c-b": { id: "c-b" } } }));
    const r = (await loadRegistry(dir))!;
    expect(r.active).toBe("");
    expect(Object.keys(r.convs)).toEqual(["c-a"]);
  });

  it("rejects an index with no valid entries", async () => {
    fs.writeFileSync(path.join(dir, "index.json"), JSON.stringify({ v: 1, active: "c-a", convs: { "c-a": { id: "c-a" } } }));
    expect(await loadRegistry(dir)).toBeNull();
  });

  it("keeps an entry with odd field values, coerced to safe ones", async () => {
    const good = blankEntry("c-a", "2026-10-20T08:00:00.000Z");
    fs.writeFileSync(
      path.join(dir, "index.json"),
      JSON.stringify({ v: 1, active: "c-a", convs: { "c-a": good, "c-b": { ...good, id: "c-b", costUsd: "1", sessionId: "s-b" }, "c-c": { ...good, id: "c-c", readOnly: "from-a-newer-build" } } }),
    );
    const r = (await loadRegistry(dir))!;
    expect(Object.keys(r.convs)).toEqual(["c-a", "c-b", "c-c"]);
    expect(r.convs["c-b"]).toMatchObject({ costUsd: null, sessionId: "s-b" });
    expect(r.convs["c-c"].readOnly).toBe("legacy");
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

  it("recovers conversations whole when they saved their own entry, and picks the newest to continue", async () => {
    const old = { ...blankEntry("c-old", "2026-10-01T00:00:00.000Z"), title: "old", sessionId: "s-old", updatedAt: "2026-10-02T00:00:00.000Z" };
    const live = { ...blankEntry("c-live", "2026-10-03T00:00:00.000Z"), title: "live", sessionId: "s-live", updatedAt: "2026-10-04T00:00:00.000Z", turns: 2 };
    await saveRegistry(dir, { v: 1, active: "c-live", convs: { "c-old": old, "c-live": live } }, ["c-old", "c-live"]);
    for (const f of fs.readdirSync(dir).filter((n) => n.startsWith("index.json"))) fs.rmSync(path.join(dir, f));
    const r = await rebuildRegistry(dir, NOW);
    expect(r.active).toBe("c-live");
    expect(r.convs["c-live"]).toMatchObject({ sessionId: "s-live", turns: 2, readOnly: null, archived: false });
    expect(r.convs["c-old"]).toMatchObject({ sessionId: "s-old", archived: false });
  });
});

describe("entry files and backups", () => {
  it("sets an unreadable index aside before replacing it", async () => {
    await saveRegistry(dir, { v: 1, active: "c-a", convs: { "c-a": blankEntry("c-a", "2026-10-20T08:00:00.000Z") } });
    fs.writeFileSync(path.join(dir, "index.json"), JSON.stringify({ v: 2, active: "c-a", convs: {} }));
    await saveRegistry(dir, { v: 1, active: "c-a", convs: { "c-a": blankEntry("c-a", "2026-10-20T08:00:00.000Z") } });
    expect(JSON.parse(fs.readFileSync(path.join(dir, "index.json.corrupt"), "utf8")).v).toBe(2);
  });

  const reg = (title: string): Registry => ({ v: 1, active: "c-a", convs: { "c-a": { ...blankEntry("c-a", "2026-10-20T08:00:00.000Z"), title } } });

  it("writes only the entries it is told changed, and adoptEntries prefers them over the index", async () => {
    await saveRegistry(dir, reg("one"), ["c-a"]);
    const stale = reg("stale").convs;
    await adoptEntries(dir, stale);
    expect(stale["c-a"].title).toBe("one");
    const other = { "c-b": blankEntry("c-b", "2026-10-20T08:00:00.000Z") };
    await saveRegistry(dir, { v: 1, active: "c-b", convs: other });
    expect(fs.existsSync(path.join(dir, "conv", "c-b", "entry.json"))).toBe(false);
  });

  it("does not rewrite an index that has not changed", async () => {
    await saveRegistry(dir, reg("one"));
    const file = path.join(dir, "index.json");
    const old = new Date(Date.now() - 3_600_000);
    fs.utimesSync(file, old, old);
    await saveRegistry(dir, reg("one"));
    expect(fs.statSync(file).mtimeMs).toBeLessThan(Date.now() - 3_000_000);
    await saveRegistry(dir, reg("two"));
    expect(JSON.parse(fs.readFileSync(file, "utf8")).convs["c-a"].title).toBe("two");
  });

  it("ignores an unreadable or mismatched entry file", async () => {
    fs.mkdirSync(path.join(dir, "conv", "c-a"), { recursive: true });
    fs.writeFileSync(path.join(dir, "conv", "c-a", "entry.json"), "{ nope");
    fs.mkdirSync(path.join(dir, "conv", "c-b"), { recursive: true });
    fs.writeFileSync(path.join(dir, "conv", "c-b", "entry.json"), JSON.stringify(blankEntry("c-other", "2026-10-20T08:00:00.000Z")));
    const convs = {};
    await adoptEntries(dir, convs);
    expect(convs).toEqual({});
  });

  it("keeps a copy from before today's first save, and falls back to it", async () => {
    await saveRegistry(dir, reg("one"));
    await saveRegistry(dir, reg("two"));
    await saveRegistry(dir, reg("three"));
    const daily = path.join(dir, "index.json.daily");
    expect(JSON.parse(fs.readFileSync(daily, "utf8")).convs["c-a"].title).toBe("one"); // not refreshed again the same day
    const yesterday = new Date(Date.now() - 86_400_000);
    fs.utimesSync(daily, yesterday, yesterday);
    await saveRegistry(dir, reg("four"));
    expect(JSON.parse(fs.readFileSync(daily, "utf8")).convs["c-a"].title).toBe("three");
    fs.writeFileSync(path.join(dir, "index.json"), "garbage");
    fs.writeFileSync(path.join(dir, "index.json.bak"), "garbage");
    expect((await loadRegistry(dir))!.convs["c-a"].title).toBe("three");
  });
});

describe("small helpers", () => {
  it("cleans a title to one line", () => {
    expect(cleanTitle(" a\n b\t\u0007c ")).toBe("a b c");
  });
  it("calls only an unused, untitled, unseeded conversation empty", () => {
    const e = blankEntry("c-a", "2026-10-20T08:00:00.000Z");
    expect(isEmptyConv(e)).toBe(true);
    expect(isEmptyConv({ ...e, seededFrom: "c-b" })).toBe(false);
    expect(isEmptyConv({ ...e, title: "t" })).toBe(false);
    expect(isEmptyConv({ ...e, readOnly: "legacy" })).toBe(false);
  });
  it("picks the newest conversation that can be continued", () => {
    const at = (id: string, updatedAt: string, over = {}) => ({ ...blankEntry(id, updatedAt), updatedAt, ...over });
    const convs = { "c-a": at("c-a", "2026-10-01T00:00:00.000Z"), "c-b": at("c-b", "2026-10-03T00:00:00.000Z", { archived: true }), "c-c": at("c-c", "2026-10-02T00:00:00.000Z") };
    expect(pickContinuable(convs)?.id).toBe("c-c");
    expect(pickContinuable({})).toBeUndefined();
  });
});

describe("buildDigest", () => {
  const ev = (seq: number, kind: string, text: string) => ({ kind, text, seq, at: "2026-10-20T08:00:00.000Z" }) as Parameters<typeof buildDigest>[0][number];

  it("keeps the words of both sides in order and leaves out tool noise", () => {
    const events = [ev(1, "user", "list workers"), { kind: "tool-use", tool: "t", toolUseId: "1", input: {}, seq: 2, at: "x" } as never, ev(3, "text", "None running.")];
    expect(buildDigest(events)).toBe("Owner: list workers\nChief: None running.");
  });

  it("keeps the newest turns when over the cap, never splitting a line", () => {
    const events = [ev(1, "user", "a".repeat(30)), ev(2, "text", "b".repeat(30)), ev(3, "user", "c".repeat(30))];
    expect(buildDigest(events, 50)).toBe(`Owner: ${"c".repeat(30)}`);
  });

  it("keeps the tail of a single line that is longer than the cap", () => {
    expect(buildDigest([ev(1, "user", "x".repeat(100))], 20)).toHaveLength(20);
  });

  it("is empty when nobody said anything", () => {
    expect(buildDigest([])).toBe("");
  });
});

describe("selectSweep", () => {
  const DAY = 86_400_000;
  const NOW = Date.parse("2026-10-27T00:00:00.000Z");
  const at = (days: number) => new Date(NOW - days * DAY).toISOString();
  const e = (id: string, days: number, archivedDays: number | null = null) => ({ ...blankEntry(id, at(days)), updatedAt: at(days), archived: archivedDays !== null, archivedAt: archivedDays === null ? null : at(archivedDays) });
  const none = { archive: [], purge: [], stamp: [] };

  it("archives idle conversations past the limit, never the active one", () => {
    const r = selectSweep([e("c-a", 40), e("c-b", 10), e("c-active", 99)], "c-active", NOW, 30, 0);
    expect(r).toEqual({ ...none, archive: ["c-a"] });
  });

  it("purges by the date of archiving, not of the last message, and only when purge is on", () => {
    const rows = [e("c-a", 100, 95), e("c-b", 100, 5), e("c-c", 100)];
    expect(selectSweep(rows, "c-x", NOW, 30, 90)).toEqual({ archive: ["c-c"], purge: ["c-a"], stamp: [] });
    expect(selectSweep(rows, "c-x", NOW, 30, 0).purge).toEqual([]);
  });

  it("asks for a date on an archived conversation that has none, instead of deleting it", () => {
    expect(selectSweep([e("c-a", 500, 0), { ...e("c-b", 500, 0), archivedAt: null }], "c-x", NOW, 0, 90)).toEqual({ ...none, stamp: ["c-b"] });
  });

  it("does nothing when both limits are off", () => {
    expect(selectSweep([e("c-a", 999)], "c-x", NOW, 0, 0)).toEqual(none);
  });

  it("archives a conversation whose last-message date is unreadable, and restarts the purge clock on an unreadable archive date", () => {
    expect(selectSweep([{ ...e("c-a", 1), updatedAt: "garbage" }], "c-x", NOW, 30, 90)).toEqual({ ...none, archive: ["c-a"] });
    expect(selectSweep([{ ...e("c-a", 500, 0), archivedAt: "garbage" }], "c-x", NOW, 0, 90)).toEqual({ ...none, stamp: ["c-a"] });
  });

  it("keeps an archived conversation that an unsent conversation was seeded from", () => {
    const seeded = { ...e("c-new", 1), turns: 0, seededFrom: "c-a" };
    expect(selectSweep([e("c-a", 100, 95), seeded], "c-x", NOW, 0, 90).purge).toEqual([]);
    expect(selectSweep([e("c-a", 100, 95), { ...seeded, turns: 2 }], "c-x", NOW, 0, 90).purge).toEqual(["c-a"]);
  });
});
