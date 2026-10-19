import fs from "node:fs";
import path from "node:path";
import type { RunFiles, RunResult, RunRow, RunsResponse } from "../../shared/types";
import { classifyName, deriveStatus, parseResult } from "./runParse";

export const MAX_RUN_DIRS = 500;
export const SCAN_CACHE_MS = 10_000;
export const MAX_RESULT_BYTES = 1_000_000;

const F = { best: "best.pt", last: "last.pt", results: "results.json", reeval: "results.inference.json", wandbId: "wandb_id.txt", v1: "PIPELINE_V1" } as const;
const EPOCH_SNAPSHOT = /^epoch(\d+)\.pt$/;

/** lstat, so a symlink is never a regular file here and is never followed. */
function regularFile(p: string): fs.Stats | null {
  try {
    const st = fs.lstatSync(p);
    return st.isFile() ? st : null;
  } catch {
    return null;
  }
}

function readResult(file: string, label: string): { value: RunResult | null; error: string | null } {
  const st = regularFile(file);
  if (!st) return { value: null, error: null };
  if (st.size > MAX_RESULT_BYTES) return { value: null, error: `${label}: larger than the size limit` };
  let text: string;
  try {
    text = fs.readFileSync(file, "utf8");
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

function buildRow(root: string, name: string): RunRow | null {
  const dir = path.join(root, name);
  let real: string;
  try {
    real = fs.realpathSync(dir);
  } catch {
    return null;
  }
  if (path.dirname(real) !== root) return null; // a directory that resolves outside the tree

  const stats = {
    bestPt: regularFile(path.join(dir, F.best)),
    lastPt: regularFile(path.join(dir, F.last)),
    resultsJson: regularFile(path.join(dir, F.results)),
    resultsInferenceJson: regularFile(path.join(dir, F.reeval)),
    wandbIdTxt: regularFile(path.join(dir, F.wandbId)),
  };
  let newestEpoch: number | null = null;
  let snapshots = 0;
  try {
    for (const entry of fs.readdirSync(dir)) {
      const m = EPOCH_SNAPSHOT.exec(entry);
      if (m && regularFile(path.join(dir, entry))) {
        snapshots++;
        newestEpoch = Math.max(newestEpoch ?? 0, Number(m[1]));
      }
    }
  } catch {
    return null;
  }
  const anyCheckpoint = Boolean(stats.bestPt || stats.lastPt || snapshots > 0);
  if (!anyCheckpoint && !stats.resultsJson && !stats.resultsInferenceJson) return null; // not a run

  const mtimes = Object.values(stats).filter((s): s is fs.Stats => s !== null).map((s) => s.mtimeMs);
  const files: RunFiles = {
    bestPt: stats.bestPt !== null,
    lastPt: stats.lastPt !== null,
    resultsJson: stats.resultsJson !== null,
    resultsInferenceJson: stats.resultsInferenceJson !== null,
    wandbIdTxt: stats.wandbIdTxt !== null,
    newestMtime: mtimes.length ? new Date(Math.max(...mtimes)).toISOString() : null,
  };

  const { kind, generation } = classifyName(name);
  const main = readResult(path.join(dir, F.results), "results.json");
  const reeval = readResult(path.join(dir, F.reeval), "results.inference.json");
  const status = deriveStatus(kind, { resultsJson: files.resultsJson, anyCheckpoint }, main.value, main.error !== null);

  const tc = main.value?.inference;
  let epochs: RunRow["epochs"] = null;
  if (tc && (tc.lastLoggedEpoch !== null || tc.configuredEpochs !== null)) epochs = { last: tc.lastLoggedEpoch, configured: tc.configuredEpochs };
  else if (kind === "ssl-pretrain" && newestEpoch !== null) epochs = { last: newestEpoch, configured: null };

  return {
    name, kind, generation, archivedV1: regularFile(path.join(dir, F.v1)) !== null, status, files,
    result: main.value, reevaluation: reeval.value, epochs,
    error: [main.error, reeval.error].filter(Boolean).join("; ") || null,
  };
}

export function scanRuns(checkpointsDir: string, now: Date = new Date()): RunsResponse {
  const scannedAt = now.toISOString();
  let root: string;
  try {
    root = fs.realpathSync(checkpointsDir);
  } catch {
    return { runs: [], scannedAt, checkpointsDirExists: false, truncated: false, error: null };
  }
  let names: string[];
  try {
    // Dirent.isDirectory() is false for a symlink, so linked directories are skipped here.
    names = fs.readdirSync(root, { withFileTypes: true }).filter((e) => e.isDirectory()).map((e) => e.name).sort();
  } catch {
    return { runs: [], scannedAt, checkpointsDirExists: true, truncated: false, error: "the checkpoints directory could not be read" };
  }
  const truncated = names.length > MAX_RUN_DIRS;
  const runs = names.slice(0, MAX_RUN_DIRS).map((n) => buildRow(root, n)).filter((r): r is RunRow => r !== null);
  return { runs, scannedAt, checkpointsDirExists: true, truncated, error: null };
}

export interface RunScanner {
  scan(): RunsResponse;
}

export function createRunScanner(checkpointsDir: string, opts: { ttlMs?: number; now?: () => number } = {}): RunScanner {
  const ttl = opts.ttlMs ?? SCAN_CACHE_MS;
  const clock = opts.now ?? Date.now;
  let cached: { at: number; value: RunsResponse } | null = null;
  return {
    scan() {
      const t = clock();
      if (cached && t - cached.at < ttl) return cached.value;
      const value = scanRuns(checkpointsDir, new Date(t));
      cached = { at: t, value };
      return value;
    },
  };
}
