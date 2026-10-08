export interface PrSummary {
  number: number; title: string; state: "OPEN" | "MERGED" | "CLOSED";
  mergedAt: string | null; headRefName: string; baseRefName: string;
  additions: number; deletions: number; changedFiles: number; url: string;
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
export interface Board {
  columns: Record<TaskStatus, Task[]>;
  conflicts: string[]; lastSyncAt: string | null; syncError: string | null;
  /** False until the first import has been run from the CLI; the board is then read from the markdown only. */
  imported: boolean;
}

export type DashEvent =
  | { type: "invalidate"; resource: "prs" | "issues" | "ci" | "tests" | "kanban" }
  | { type: "tests-line"; line: string };
