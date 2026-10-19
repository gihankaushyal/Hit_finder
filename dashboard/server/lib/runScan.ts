import type { Stats } from "node:fs";
import fs from "node:fs/promises";
import path from "node:path";
import type { RunFiles, RunResult, RunRow, RunsResponse } from "../../shared/types";
import { classifyName, deriveStatus, parseResult } from "./runParse";

export const MAX_RUN_DIRS = 500;
export const SCAN_CACHE_MS = 10_000;
export const MAX_RESULT_BYTES = 1_000_000;
/** Directories inspected at once: enough to hide filesystem latency, few enough not to flood a shared one. */
export const SCAN_CONCURRENCY = 8;

const F = { best: "best.pt", last: "last.pt", results: "results.json", reeval: "results.inference.json", wandbId: "wandb_id.txt", v1: "PIPELINE_V1" } as const;
const EPOCH_SNAPSHOT = /^epoch(\d+)\.pt$/;

/** lstat, so a symlink is never a regular file here and is never followed. */
async function regularFile(p: string): Promise<Stats | null> {
  try {
    const st = await fs.lstat(p);
    return st.isFile() ? st : null;
  } catch {
    return null;
  }
}

async function readResult(file: string, label: string): Promise<{ value: RunResult | null; error: string | null }> {
  const st = await regularFile(file);
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

async function buildRow(root: string, name: string): Promise<RunRow | null> {
  const dir = path.join(root, name);
  let real: string;
  try {
    real = await fs.realpath(dir);
  } catch {
    return null;
  }
  if (path.dirname(real) !== root) return null; // a directory that resolves outside the tree

  const [bestPt, lastPt, resultsJson, resultsInferenceJson, wandbIdTxt, v1Marker] = await Promise.all(
    [F.best, F.last, F.results, F.reeval, F.wandbId, F.v1].map((f) => regularFile(path.join(dir, f))),
  );
  let newestEpoch: number | null = null;
  let snapshots = 0;
  try {
    for (const entry of await fs.readdir(dir)) {
      const m = EPOCH_SNAPSHOT.exec(entry);
      if (m && (await regularFile(path.join(dir, entry)))) {
        snapshots++;
        newestEpoch = Math.max(newestEpoch ?? 0, Number(m[1]));
      }
    }
  } catch {
    return null;
  }
  const anyCheckpoint = Boolean(bestPt || lastPt || snapshots > 0);
  if (!anyCheckpoint && !resultsJson && !resultsInferenceJson) return null; // not a run

  const mtimes = [bestPt, lastPt, resultsJson, resultsInferenceJson, wandbIdTxt].filter((s): s is Stats => s !== null).map((s) => s.mtimeMs);
  const files: RunFiles = {
    bestPt: bestPt !== null,
    lastPt: lastPt !== null,
    resultsJson: resultsJson !== null,
    resultsInferenceJson: resultsInferenceJson !== null,
    wandbIdTxt: wandbIdTxt !== null,
    newestMtime: mtimes.length ? new Date(Math.max(...mtimes)).toISOString() : null,
  };

  const { kind, generation } = classifyName(name);
  const [main, reeval] = await Promise.all([readResult(path.join(dir, F.results), "results.json"), readResult(path.join(dir, F.reeval), "results.inference.json")]);
  // Old-style pretrain directories have a legacy name but the same shape: snapshots, no best.pt, no results.
  const pretrainShaped = !files.bestPt && (files.lastPt || snapshots > 0) && !files.resultsJson && !files.resultsInferenceJson;
  const status = deriveStatus(kind, { resultsJson: files.resultsJson, anyCheckpoint, pretrainShaped }, main.value, main.error !== null);

  const tc = main.value?.inference;
  let epochs: RunRow["epochs"] = null;
  if (tc && (tc.lastLoggedEpoch !== null || tc.configuredEpochs !== null)) epochs = { last: tc.lastLoggedEpoch, configured: tc.configuredEpochs };
  else if (status === "pretrain" && newestEpoch !== null) epochs = { last: newestEpoch, configured: null };

  return {
    name, kind, generation, archivedV1: v1Marker !== null, status, files,
    result: main.value, reevaluation: reeval.value, epochs,
    error: [main.error, reeval.error].filter(Boolean).join("; ") || null,
  };
}

export async function scanRuns(checkpointsDir: string, now: Date = new Date()): Promise<RunsResponse> {
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
    names = (await fs.readdir(root, { withFileTypes: true })).filter((e) => e.isDirectory()).map((e) => e.name).sort();
  } catch {
    return { runs: [], scannedAt, checkpointsDirExists: true, truncated: false, error: "the checkpoints directory could not be read" };
  }
  const truncated = names.length > MAX_RUN_DIRS;
  const rows = await mapLimit(names.slice(0, MAX_RUN_DIRS), SCAN_CONCURRENCY, (n) => buildRow(root, n));
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
