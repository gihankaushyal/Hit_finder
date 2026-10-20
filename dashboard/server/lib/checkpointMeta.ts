import { constants, type Stats } from "node:fs";
import fs from "node:fs/promises";
import { readPickleTopLevel } from "./pickleScan";

/**
 * Reads the epoch (and validation F1) stored in a torch checkpoint without loading the model.
 *
 * A torch checkpoint is an uncompressed zip whose `data.pkl` entry is a small pickle holding the
 * top-level dict. Only that entry is read: the zip directory at the end of the file, then the entry's
 * bytes (tens of kilobytes out of hundreds of megabytes). The pickle is parsed by `pickleScan`, which
 * never runs anything in it. Every failure, including an unsupported layout, is `null`.
 */

// Real checkpoints: a data.pkl of 35-100 KB and a zip directory of tens of KB. These caps are generous bounds.
export const MAX_PICKLE_BYTES = 4 * 1024 * 1024;
const MAX_CENTRAL_DIRECTORY_BYTES = 2 * 1024 * 1024;
const MAX_ENTRIES = 100_000;
const EOCD_SIGNATURE = 0x06054b50;
const CENTRAL_SIGNATURE = 0x02014b50;
const LOCAL_SIGNATURE = 0x04034b50;
const EOCD_MIN = 22;
const EOCD_SEARCH = 65_535 + EOCD_MIN;
const ZIP32_LIMIT = 0xffffffff;
const STORED = 0;

export interface CheckpointMeta {
  epoch: number;
}

async function readAt(handle: fs.FileHandle, position: number, length: number): Promise<Buffer | null> {
  const buf = Buffer.alloc(length);
  let got = 0;
  while (got < length) {
    const { bytesRead } = await handle.read(buf, got, length - got, position + got);
    if (bytesRead === 0) return null;
    got += bytesRead;
  }
  return buf;
}

/** The bytes of the stored `data.pkl` entry, or null if the file is not a layout this reader supports. */
async function readDataPkl(handle: fs.FileHandle, size: number): Promise<Buffer | null> {
  const tailLength = Math.min(size, EOCD_SEARCH);
  const tail = await readAt(handle, size - tailLength, tailLength);
  if (!tail) return null;
  let eocd = -1;
  for (let i = tail.length - EOCD_MIN; i >= 0; i--) {
    if (tail.readUInt32LE(i) === EOCD_SIGNATURE) {
      eocd = i;
      break;
    }
  }
  if (eocd < 0) return null;
  const entries = tail.readUInt16LE(eocd + 10);
  const directorySize = tail.readUInt32LE(eocd + 12);
  const directoryOffset = tail.readUInt32LE(eocd + 16);
  // zip64 (archives over 4 GB) is not supported: the checkpoints here are a few hundred megabytes.
  if (entries === 0xffff || directorySize === ZIP32_LIMIT || directoryOffset === ZIP32_LIMIT) return null;
  if (directorySize === 0 || directorySize > MAX_CENTRAL_DIRECTORY_BYTES || directoryOffset + directorySize > size) return null;
  const directory = await readAt(handle, directoryOffset, directorySize);
  if (!directory) return null;

  let at = 0;
  for (let n = 0; n < Math.min(entries, MAX_ENTRIES) && at + 46 <= directory.length; n++) {
    if (directory.readUInt32LE(at) !== CENTRAL_SIGNATURE) return null;
    const method = directory.readUInt16LE(at + 10);
    const compressed = directory.readUInt32LE(at + 20);
    const uncompressed = directory.readUInt32LE(at + 24);
    const nameLength = directory.readUInt16LE(at + 28);
    const extraLength = directory.readUInt16LE(at + 30);
    const commentLength = directory.readUInt16LE(at + 32);
    const localOffset = directory.readUInt32LE(at + 42);
    if (at + 46 + nameLength > directory.length) return null;
    const name = directory.subarray(at + 46, at + 46 + nameLength).toString("utf8");
    if (name === "data.pkl" || name.endsWith("/data.pkl")) {
      if (method !== STORED || compressed !== uncompressed || uncompressed > MAX_PICKLE_BYTES) return null;
      const local = await readAt(handle, localOffset, 30);
      if (!local || local.readUInt32LE(0) !== LOCAL_SIGNATURE) return null;
      const start = localOffset + 30 + local.readUInt16LE(26) + local.readUInt16LE(28);
      if (start + uncompressed > size) return null;
      return readAt(handle, start, uncompressed);
    }
    at += 46 + nameLength + extraLength + commentLength;
  }
  return null;
}

const keyOf = (st: Stats): string => `${st.size}:${st.mtimeMs}`;

/** What was read, and the size and mtime of the very file it was read from (not of an earlier look at it). */
async function readChecked(file: string): Promise<{ meta: CheckpointMeta | null; key: string } | null> {
  let handle: fs.FileHandle | undefined;
  try {
    // O_NOFOLLOW: a symlink is never opened, so a link cannot point this reader outside the tree.
    handle = await fs.open(file, constants.O_RDONLY | constants.O_NOFOLLOW);
    const st = await handle.stat();
    const key = keyOf(st);
    if (!st.isFile() || st.size < EOCD_MIN) return { meta: null, key };
    const pickle = await readDataPkl(handle, st.size);
    const top = pickle ? readPickleTopLevel(pickle) : null;
    const epoch = top?.get("epoch");
    return { meta: typeof epoch === "number" && Number.isFinite(epoch) ? { epoch } : null, key };
  } catch {
    return null;
  } finally {
    await handle?.close().catch(() => undefined);
  }
}

export async function readCheckpointMeta(file: string): Promise<CheckpointMeta | null> {
  return (await readChecked(file))?.meta ?? null;
}

export interface CheckpointMetaReader {
  /** `stats` are the caller's lstat of `file`, if it has them, so the file is not looked up twice. */
  read(file: string, stats?: Stats | null): Promise<CheckpointMeta | null>;
  size(): number;
}

/**
 * Remembers what it read, keyed by size and modification time, so a rescan every few seconds does not
 * go back to a network filesystem for a checkpoint that has not changed. The key stored is that of the
 * file actually read, so a checkpoint rewritten between the caller's look and the read is never served
 * stale. Unreadable files are remembered too. Least recently used entries are dropped first.
 */
export function createCheckpointMetaReader(opts: { maxEntries?: number } = {}): CheckpointMetaReader {
  const max = opts.maxEntries ?? 2000;
  const cache = new Map<string, { key: string; meta: CheckpointMeta | null }>();
  return {
    async read(file, stats) {
      let st = stats ?? null;
      if (!st) {
        try {
          st = await fs.lstat(file);
        } catch {
          return null;
        }
      }
      if (!st.isFile()) return null;
      const hit = cache.get(file);
      if (hit && hit.key === keyOf(st)) {
        cache.delete(file); // re-insert: most recently used last
        cache.set(file, hit);
        return hit.meta;
      }
      const read = await readChecked(file);
      const entry = { key: read?.key ?? keyOf(st), meta: read?.meta ?? null };
      cache.delete(file);
      cache.set(file, entry);
      while (cache.size > max) cache.delete(cache.keys().next().value as string);
      return entry.meta;
    },
    size: () => cache.size,
  };
}
