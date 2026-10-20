import fs from "node:fs";
import fsp from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { MAX_PICKLE_BYTES, createCheckpointMetaReader, readCheckpointMeta } from "../server/lib/checkpointMeta";

const FIX = path.join(__dirname, "fixtures", "ckpt");
let tmp: string;
beforeEach(() => { tmp = fs.mkdtempSync(path.join(os.tmpdir(), "ckptmeta-")); });
afterEach(() => { vi.restoreAllMocks(); fs.rmSync(tmp, { recursive: true, force: true }); });
const copy = (name: string, as = name) => { const p = path.join(tmp, as); fs.copyFileSync(path.join(FIX, name), p); return p; };

describe("readCheckpointMeta", () => {
  it("reads the epoch and validation F1 from a real torch checkpoint without loading the model", async () => {
    expect(await readCheckpointMeta(path.join(FIX, "best-187.pt"))).toEqual({ epoch: 187, valF1: 0.93 });
    expect(await readCheckpointMeta(path.join(FIX, "best-5.pt"))).toEqual({ epoch: 5, valF1: 0.5 });
  });

  it("a checkpoint with no val_f1 (a pretrain snapshot) has valF1 null", async () => {
    expect(await readCheckpointMeta(path.join(FIX, "last-400.pt"))).toEqual({ epoch: 400, valF1: null });
  });

  it("returns null for a missing file, a directory, an empty file and random bytes", async () => {
    expect(await readCheckpointMeta(path.join(tmp, "nope.pt"))).toBeNull();
    expect(await readCheckpointMeta(tmp)).toBeNull();
    fs.writeFileSync(path.join(tmp, "empty.pt"), "");
    expect(await readCheckpointMeta(path.join(tmp, "empty.pt"))).toBeNull();
    fs.writeFileSync(path.join(tmp, "junk.pt"), Buffer.alloc(5000, 7));
    expect(await readCheckpointMeta(path.join(tmp, "junk.pt"))).toBeNull();
  });

  it("returns null for a truncated checkpoint (no zip directory at the end)", async () => {
    const whole = fs.readFileSync(path.join(FIX, "best-187.pt"));
    fs.writeFileSync(path.join(tmp, "cut.pt"), whole.subarray(0, whole.length >> 1));
    expect(await readCheckpointMeta(path.join(tmp, "cut.pt"))).toBeNull();
  });

  it("does not follow a symlink", async () => {
    fs.symlinkSync(path.join(FIX, "best-187.pt"), path.join(tmp, "link.pt"));
    expect(await readCheckpointMeta(path.join(tmp, "link.pt"))).toBeNull();
  });

  /** Offset of the central-directory entry whose name ends with data.pkl. */
  const dataPklEntry = (buf: Buffer) => {
    for (let i = 0; i + 46 < buf.length; i++) {
      if (buf.readUInt32LE(i) === 0x02014b50) {
        const n = buf.readUInt16LE(i + 28);
        if (buf.subarray(i + 46, i + 46 + n).toString().endsWith("data.pkl")) return i;
      }
    }
    throw new Error("no data.pkl entry");
  };

  it("refuses a compressed data.pkl, because only stored entries are read", async () => {
    const p = copy("best-187.pt", "deflated.pt");
    const buf = fs.readFileSync(p);
    buf.writeUInt16LE(8, dataPklEntry(buf) + 10); // method: deflate
    fs.writeFileSync(p, buf);
    expect(await readCheckpointMeta(p)).toBeNull();
  });

  it("refuses an oversized data.pkl without reading it", async () => {
    const p = copy("best-187.pt", "huge.pt");
    const buf = fs.readFileSync(p);
    const at = dataPklEntry(buf);
    buf.writeUInt32LE(MAX_PICKLE_BYTES + 1, at + 20);
    buf.writeUInt32LE(MAX_PICKLE_BYTES + 1, at + 24);
    fs.writeFileSync(p, buf);
    expect(await readCheckpointMeta(p)).toBeNull();
  });

  it("never throws, whatever the bytes", async () => {
    const whole = fs.readFileSync(path.join(FIX, "best-187.pt"));
    for (let seed = 1; seed <= 25; seed++) {
      const b = Buffer.from(whole);
      for (let k = 0; k < 6; k++) b[(seed * 7919 + k * 104729) % b.length] ^= 0xff;
      fs.writeFileSync(path.join(tmp, "fuzz.pt"), b);
      await expect(readCheckpointMeta(path.join(tmp, "fuzz.pt"))).resolves.toSatisfy((v: unknown) => v === null || typeof v === "object");
    }
  });
});

describe("createCheckpointMetaReader", () => {
  it("reads a file once while it is unchanged, and again when it changes", async () => {
    const open = vi.spyOn(fsp, "open");
    const reader = createCheckpointMetaReader();
    const p = copy("best-187.pt", "best.pt");
    expect(await reader.read(p)).toEqual({ epoch: 187, valF1: 0.93 });
    const afterFirst = open.mock.calls.length;
    expect(afterFirst).toBeGreaterThan(0);
    expect(await reader.read(p)).toEqual({ epoch: 187, valF1: 0.93 });
    expect(open.mock.calls.length).toBe(afterFirst); // served from the cache
    fs.copyFileSync(path.join(FIX, "best-5.pt"), p); // different bytes and mtime
    expect(await reader.read(p)).toEqual({ epoch: 5, valF1: 0.5 });
    expect(open.mock.calls.length).toBeGreaterThan(afterFirst);
  });

  it("caches an unreadable result too, so a bad file is not retried every scan", async () => {
    const open = vi.spyOn(fsp, "open");
    const reader = createCheckpointMetaReader();
    const p = path.join(tmp, "junk.pt");
    fs.writeFileSync(p, Buffer.alloc(5000, 7));
    expect(await reader.read(p)).toBeNull();
    const n = open.mock.calls.length;
    expect(await reader.read(p)).toBeNull();
    expect(open.mock.calls.length).toBe(n);
  });

  it("does not grow without bound", async () => {
    const reader = createCheckpointMetaReader({ maxEntries: 3 });
    const paths = [1, 2, 3, 4, 5].map((i) => copy("best-5.pt", `b${i}.pt`));
    for (const p of paths) await reader.read(p);
    expect(reader.size()).toBeLessThanOrEqual(3);
  });
});
