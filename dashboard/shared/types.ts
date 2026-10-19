export interface PrChecks {
  state: "passing" | "failing" | "pending" | "none";
  passing: number; failing: number; pending: number;
}
export interface PrSummary {
  number: number; title: string; state: "OPEN" | "MERGED" | "CLOSED";
  mergedAt: string | null; headRefName: string; baseRefName: string;
  additions: number; deletions: number; changedFiles: number; url: string;
  /** Check state, present on open PRs only. */
  checks?: PrChecks;
}
export interface PrDetail extends PrSummary {
  summary: string[];                                  // bullets under "## Summary"
  testPlan: { text: string; checked: boolean }[];     // items under "## Test plan"
  files: { path: string; additions: number; deletions: number }[];
}
export interface PrsResponse { latest: PrDetail | null; recent: PrSummary[]; open: PrSummary[] }

export interface Issue {
  number: number; title: string; labels: string[];
  createdAt: string; updatedAt: string; url: string;
}

export interface CiRun {
  id: number; status: string; conclusion: string | null;
  branch: string; event: string; createdAt: string; url: string;
}

export interface JunitSummary {
  tests: number; passed: number; failed: number; errors: number; skipped: number;
  durationSec: number; failures: { name: string; message: string }[];
}
export interface TestRunState {
  status: "idle" | "running" | "done" | "error";
  startedAt: string | null; finishedAt: string | null; commit: string | null;
  summary: JunitSummary | null; tail: string[]; error: string | null;
}

export type TaskStatus = "todo" | "in-progress" | "blocked" | "done";
export interface Task {
  number: number; title: string; status: TaskStatus; kind: string | null;
  url: string; updatedAt: string; inFile: boolean;
}
export interface BoardConflict { issue: number | null; message: string }
export interface Board {
  columns: Record<TaskStatus, Task[]>;
  conflicts: string[];
  /** The same conflicts as `conflicts`, in the same order, with the issue each one concerns (null when none). */
  conflictItems?: BoardConflict[];
  lastSyncAt: string | null; syncError: string | null;
  /** False until the first import has been run from the CLI; the board is then read from the markdown only. */
  imported: boolean;
}

export type DashEvent =
  | { type: "invalidate"; resource: "prs" | "issues" | "ci" | "tests" | "kanban" }
  | { type: "tests-line"; line: string };

export type RunKind = "track1" | "ssl-pretrain" | "ssl-finetune" | "ssl-probe" | "legacy";
/** pretrain: SSL pretraining directories never write results.json; their progress is the newest epoch snapshot. */
export type RunStatus = "complete" | "provisional" | "unknown" | "no-results" | "pretrain";
export type TrainingCheckStatus = "complete" | "incomplete" | "unknown";
export type MetricBlock = Record<string, number | null>;

export interface RunInference {
  aggregation: string | null;
  checkpointEpoch: number | null;
  evaluatedAt: string | null;
  trainingStatus: TrainingCheckStatus | null;
  configuredEpochs: number | null;
  lastLoggedEpoch: number | null;
  wandbState: string | null;
  /** Shown verbatim to the user. */
  warnings: string[];
}

export interface RunResult {
  foldId: number | null;
  testDetector: string | null;
  inferenceThreshold: number | null;
  cross: MetricBlock | null;
  inDomain: MetricBlock | null;
  track: string | null;
  probe: boolean | null;
  pretrainEpoch: number | null;
  inference: RunInference | null;
}

export interface RunFiles {
  bestPt: boolean;
  lastPt: boolean;
  resultsJson: boolean;
  resultsInferenceJson: boolean;
  wandbIdTxt: boolean;
  /** ISO time of the newest of the files above, or null when none exist. */
  newestMtime: string | null;
}

export interface RunRow {
  name: string;
  kind: RunKind;
  /** The N of `-v<N>` in a name that follows the naming rules; null for legacy names. */
  generation: number | null;
  /** A PIPELINE_V1 marker file exists. It wins over the name. */
  archivedV1: boolean;
  status: RunStatus;
  files: RunFiles;
  result: RunResult | null;
  /** The parsed results.inference.json, a re-evaluation, never the fold's headline result. */
  reevaluation: RunResult | null;
  epochs: { last: number | null; configured: number | null } | null;
  /** A fixed message when a JSON file was unreadable or corrupt; the row is still returned. */
  error: string | null;
}

export interface RunsResponse {
  runs: RunRow[];
  scannedAt: string;
  checkpointsDirExists: boolean;
  /** More run directories than the scan limit existed; the rest were not listed. */
  truncated: boolean;
  error: string | null;
}
