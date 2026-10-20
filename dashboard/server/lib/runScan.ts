import type { Dirent, Stats } from "node:fs";
import fs from "node:fs/promises";
import path from "node:path";
import type { RunFiles, RunResult, RunRow, RunsResponse } from "../../shared/types";
import { createCheckpointMetaReader, type CheckpointMetaReader } from "./checkpointMeta";
import { classifyName, deriveStatus, parseResult } from "./runParse";

export const MAX_RUN_DIRS = 500;
export const SCAN_CACHE_MS = 10_000;
export const MAX_RESULT_BYTES = 1_000_000;
/** Directories inspected at once: enough to hide filesystem latency, few enough not to flood a shared one. */
export const SCAN_CONCURRENCY = 8;

const F = { best: "best.pt", last: "last.pt", results: "results.json", reeval: "results.inference.json", wandbId: "wandb_id.txt", v1: "PIPELINE_V1" } as const;
const EPOCH_SNAPSHOT = /^epoch(\d+)\.pt$/;

/** Shared by every scan, so an unchanged best.pt is read once, not on each rescan. */
const sharedMeta = createCheckpointMetaReader();

/** lstat, so a symlink is never a regular file here and is never followed. */
async function regularFile(p: string): Promise<Stats | null> {
  try {
    const st = await fs.lstat(p);
    return st.isFile() ? st : null;
  } catch {
    return null;
  }
}

async function readResult(file: string, label: string, st: Stats | null): Promise<{ value: RunResult | null; error: string | null }> {
  if (!st) return { value: null, error: null };
  if (st.size > MAX_RESULT_BYTES) return { value: null, error: `${label}: larger than the size limit` };
  let text: string;
  try {
    text = await fs.readFile(file, "utf8");
  } catch {
    return { value: null, error: `${label}: could not be read` };
  }
  try {
    return { value: parseResult(text), error: null };
  } catch (e) {
    // parseResult only throws fixed, path-free messages.
    return { value: null, error: `${label}: ${e instanceof Error ? e.message : "could not be parsed"}` };
  }
}

async function mapLimit<T, R>(items: T[], limit: number, fn: (item: T) => Promise<R>): Promise<R[]> {
  const out = new Array<R>(items.length);
  let next = 0;
  const worker = async () => {
    for (;;) {
      const i = next++;
      if (i >= items.length) return;
      out[i] = await fn(items[i]);
    }
  };
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker));
  return out;
}

async function buildRow(root: string, name: string, meta: CheckpointMetaReader): Promise<RunRow | null> {
  const dir = path.join(root, name);
  let real: string;
  try {
    real = await fs.realpath(dir);
  } catch {
    return null;
  }
  if (path.dirname(real) !== root) return null; // a directory that resolves outside the tree

  // One readdir tells which entries are regular files (Dirent.isFile() is false for a symlink, so links are
  // never followed). Only the few files whose size or mtime is needed are stat-ed afterwards.
  let entries: Dirent[];
  try {
    entries = await fs.readdir(dir, { withFileTypes: true });
  } catch {
    return null;
  }
  const isFile = new Set(entries.filter((e) => e.isFile()).map((e) => e.name));
  let newestEpoch: number | null = null;
  let newestSnapshot: string | null = null;
  let snapshots = 0;
  for (const n of isFile) {
    const m = EPOCH_SNAPSHOT.exec(n);
    if (!m) continue;
    snapshots++;
    const epoch = Number(m[1]);
    if (newestEpoch === null || epoch > newestEpoch) {
      newestEpoch = epoch;
      newestSnapshot = n;
    }
  }
  const statIf = (name: string | null) => (name !== null && isFile.has(name) ? regularFile(path.join(dir, name)) : Promise.resolve(null));
  const [bestSt, lastSt, resultsSt, reevalSt, snapshotSt] = await Promise.all([F.best, F.last, F.results, F.reeval, newestSnapshot].map(statIf));
  const anyCheckpoint = Boolean(bestSt || lastSt || snapshots > 0);
  if (!anyCheckpoint && !resultsSt && !reevalSt) return null; // not a run

  const mtimes = [bestSt, lastSt, resultsSt, reevalSt, snapshotSt].filter((x): x is Stats => x !== null).map((x) => x.mtimeMs);
  const files: RunFiles = {
    bestPt: bestSt !== null,
    lastPt: lastSt !== null,
    resultsJson: resultsSt !== null,
    resultsInferenceJson: reevalSt !== null,
    wandbIdTxt: isFile.has(F.wandbId),
    newestMtime: mtimes.length ? new Date(Math.max(...mtimes)).toISOString() : null,
  };

  const { kind, generation } = classifyName(name);
  const [main, reeval] = await Promise.all([readResult(path.join(dir, F.results), "results.json", resultsSt), readResult(path.join(dir, F.reeval), "results.inference.json", reevalSt)]);
  // Old-style pretrain directories have an unrecognised (legacy) name but the same shape: snapshots, no best.pt,
  // no results. A recognised fine-tune, probe or Track 1 name in that state is simply a run without results yet.
  const pretrainShaped = kind === "legacy" && !files.bestPt && (files.lastPt || snapshots > 0) && !files.resultsJson && !files.resultsInferenceJson;
  const status = deriveStatus(kind, { resultsJson: files.resultsJson, anyCheckpoint, pretrainShaped }, main.value, main.error !== null);

  // Best source first: what the training code recorded, then an inference pass's check against W&B, then a
  // pretrain snapshot, and only last the epoch stored inside best.pt (the best epoch, not the epochs trained).
  const t = main.value?.training;
  const tc = main.value?.inference;
  let epochs: RunRow["epochs"] = null;
  if (t && (t.lastEpoch !== null || t.configuredEpochs !== null || t.bestEpoch !== null)) {
    epochs = { last: t.lastEpoch, configured: t.configuredEpochs, best: t.bestEpoch, source: "training" };
  } else if (tc && (tc.lastLoggedEpoch !== null || tc.configuredEpochs !== null)) {
    epochs = { last: tc.lastLoggedEpoch, configured: tc.configuredEpochs, best: tc.checkpointEpoch, source: "training-check" };
  } else if (status === "pretrain" && newestEpoch !== null) {
    epochs = { last: newestEpoch, configured: null, best: null, source: "snapshot" };
  } else if (files.bestPt) {
    const m = await meta.read(path.join(dir, F.best));
    if (m?.epoch != null) epochs = { last: null, configured: null, best: m.epoch, source: "checkpoint" };
  }

  return {
    name, kind, generation, archivedV1: isFile.has(F.v1), status, files,
    result: main.value, reevaluation: reeval.value, epochs,
    error: [main.error, reeval.error].filter(Boolean).join("; ") || null,
  };
}

export async function scanRuns(
  checkpointsDir: string,
  now: Date = new Date(),
  deps: { meta?: CheckpointMetaReader } = {},
): Promise<RunsResponse> {
  const meta = deps.meta ?? sharedMeta;
  const scannedAt = now.toISOString();
  let root: string;
  try {
    root = await fs.realpath(checkpointsDir);
  } catch {
    return { runs: [], scannedAt, checkpointsDirExists: false, truncated: false, error: null };
  }
  let names: string[];
  try {
    // Dirent.isDirectory() is false for a symlink, so linked directories are skipped here.
    // Numeric collation keeps -v2 before -v10.
    names = (await fs.readdir(root, { withFileTypes: true })).filter((e) => e.isDirectory()).map((e) => e.name).sort((a, b) => a.localeCompare(b, "en", { numeric: true }));
  } catch {
    return { runs: [], scannedAt, checkpointsDirExists: true, truncated: false, error: "the checkpoints directory could not be read" };
  }
  const truncated = names.length > MAX_RUN_DIRS;
  const rows = await mapLimit(names.slice(0, MAX_RUN_DIRS), SCAN_CONCURRENCY, (n) => buildRow(root, n, meta));
  return { runs: rows.filter((r): r is RunRow => r !== null), scannedAt, checkpointsDirExists: true, truncated, error: null };
}

export interface RunScanner {
  scan(): Promise<RunsResponse>;
}

export interface RunScannerOptions {
  ttlMs?: number;
  now?: () => number;
  /** Replaces the real scan; for tests. */
  scanFn?: (checkpointsDir: string, now: Date) => Promise<RunsResponse>;
}

/**
 * A scan is cached for `ttlMs`, and requests that arrive while one is running share it, so a burst of
 * requests never starts more than one walk of the filesystem. A failed scan is not cached.
 */
export function createRunScanner(checkpointsDir: string, opts: RunScannerOptions = {}): RunScanner {
  const ttl = opts.ttlMs ?? SCAN_CACHE_MS;
  const clock = opts.now ?? Date.now;
  const run = opts.scanFn ?? scanRuns;
  let cached: { at: number; value: Promise<RunsResponse> } | null = null;
  return {
    scan() {
      const t = clock();
      if (cached && t - cached.at < ttl) return cached.value;
      const value = run(checkpointsDir, new Date(t));
      const entry = { at: t, value };
      cached = entry;
      value.catch(() => {
        if (cached === entry) cached = null;
      });
      return value;
    },
  };
}
