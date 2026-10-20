import { describe, it, expect, beforeEach, afterEach } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { openJournal, JOURNAL_TRUNCATED_TEXT } from "../server/agents/journal";

let dir: string;
let file: string;
beforeEach(() => {
  dir = fs.mkdtempSync(path.join(os.tmpdir(), "dash-journal-"));
  file = path.join(dir, "a", "journal.jsonl");
});
afterEach(() => fs.rmSync(dir, { recursive: true, force: true }));

describe("journal", () => {
  it("assigns increasing sequence numbers starting at 1", async () => {
    const j = await openJournal(file);
    const a = await j.append({ kind: "text", text: "one" });
    const b = await j.append({ kind: "text", text: "two" });
    expect([a.seq, b.seq]).toEqual([1, 2]);
    expect(j.lastSeq()).toBe(2);
  });

  it("reads entries after a cursor, with a limit", async () => {
    const j = await openJournal(file);
    for (const t of ["a", "b", "c", "d"]) await j.append({ kind: "text", text: t });
    const all = await j.read(0, 100);
    expect(all.map((e) => e.seq)).toEqual([1, 2, 3, 4]);
    const rest = await j.read(2, 100);
    expect(rest.map((e) => e.seq)).toEqual([3, 4]);
    const limited = await j.read(0, 2);
    expect(limited.map((e) => e.seq)).toEqual([1, 2]);
    expect(await j.read(4, 10)).toEqual([]);
  });

  it("stamps each entry with an ISO time", async () => {
    const j = await openJournal(file);
    const e = await j.append({ kind: "user", text: "hi" });
    expect(new Date(e.at).toISOString()).toBe(e.at);
  });

  it("keeps order for concurrent appends", async () => {
    const j = await openJournal(file);
    await Promise.all(Array.from({ length: 50 }, (_, i) => j.append({ kind: "text", text: String(i) })));
    const entries = await j.read(0, 100);
    expect(entries.map((e) => e.seq)).toEqual(Array.from({ length: 50 }, (_, i) => i + 1));
    expect(entries.map((e) => (e.kind === "text" ? e.text : ""))).toEqual(Array.from({ length: 50 }, (_, i) => String(i)));
  });

  it("continues the sequence after reopening", async () => {
    const j = await openJournal(file);
    await j.append({ kind: "text", text: "x" });
    await j.append({ kind: "text", text: "y" });
    const again = await openJournal(file);
    expect(again.lastSeq()).toBe(2);
    const e = await again.append({ kind: "text", text: "z" });
    expect(e.seq).toBe(3);
  });

  it("survives a corrupt last line and does not reuse its sequence", async () => {
    const j = await openJournal(file);
    await j.append({ kind: "text", text: "ok" });
    fs.appendFileSync(file, '{"seq":2,"at":"2026-01-01T00:00:00.000Z","kind":"te');
    const again = await openJournal(file);
    expect(again.lastSeq()).toBe(1);
    const e = await again.append({ kind: "text", text: "after" });
    expect(e.seq).toBe(2);
    const entries = await again.read(0, 10);
    expect(entries.map((x) => (x.kind === "text" ? x.text : ""))).toEqual(["ok", "after"]);
  });

  it("skips a corrupt line in the middle", async () => {
    const j = await openJournal(file);
    await j.append({ kind: "text", text: "a" });
    fs.appendFileSync(file, "not json\n");
    await j.append({ kind: "text", text: "b" });
    const entries = await j.read(0, 10);
    expect(entries.map((x) => (x.kind === "text" ? x.text : ""))).toEqual(["a", "b"]);
  });

  it("stops writing at the size cap and leaves one truncation marker", async () => {
    const j = await openJournal(file, { maxBytes: 600 });
    for (let i = 0; i < 20; i++) await j.append({ kind: "text", text: "x".repeat(100) });
    const entries = await j.read(0, 100);
    const markers = entries.filter((e) => e.kind === "error" && e.message === JOURNAL_TRUNCATED_TEXT);
    expect(markers).toHaveLength(1);
    expect(entries.length).toBeLessThan(20);
    expect(fs.statSync(file).size).toBeLessThan(1200);
  });

  it("returns appended entries intact when the text contains newlines", async () => {
    const j = await openJournal(file);
    await j.append({ kind: "text", text: "line1\nline2\r\nline3" });
    const [e] = await j.read(0, 1);
    expect(e.kind === "text" && e.text).toBe("line1\nline2\r\nline3");
  });

  it("serves reads from memory instead of re-parsing the file", async () => {
    const j = await openJournal(file);
    for (let i = 0; i < 20; i++) await j.append({ kind: "text", text: String(i) });
    fs.writeFileSync(file, "");
    expect((await j.read(15, 100)).map((e) => e.seq)).toEqual([16, 17, 18, 19, 20]);
  });

  it("is empty before the first append", async () => {
    const j = await openJournal(file);
    expect(j.lastSeq()).toBe(0);
    expect(await j.read(0, 10)).toEqual([]);
  });
});
