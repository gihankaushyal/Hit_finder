import { describe, it, expect, beforeEach, afterEach } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { readJournalFile } from "../server/agents/journal";

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
