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
  | { type: "invalidate"; resource: "prs" | "issues" | "ci" | "tests" | "kanban" | "agents" | "approvals" | "chat" }
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

/** The block the training code writes into results.json: how far the run got. */
export interface RunTraining {
  configuredEpochs: number | null;
  /** Last epoch completed, as the loop counts (1-based). */
  lastEpoch: number | null;
  /** Epoch saved in best.pt; null when no epoch ever improved validation F1. */
  bestEpoch: number | null;
  earlyStopped: boolean | null;
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
  training: RunTraining | null;
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

/**
 * Where an epoch count came from, best first: `training` is the block the training code writes into
 * results.json (true epochs trained), `training-check` comes from an inference pass that asked W&B,
 * `checkpoint` is only the best epoch stored inside best.pt (epochs trained were never recorded),
 * and `snapshot` is the newest epoch<N>.pt of a pretraining run.
 */
export type EpochSource = "training" | "training-check" | "checkpoint" | "snapshot";

export interface RunEpochs {
  last: number | null;
  configured: number | null;
  best: number | null;
  source: EpochSource;
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
  epochs: RunEpochs | null;
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

// ---- Agents (milestone 2) ----

/** starting, running, waiting-approval (blocked on a card), idle (finished a turn, can take a message), then a final state. */
export type AgentState = "starting" | "running" | "waiting-approval" | "idle" | "done" | "failed" | "stopped";
export type AgentRole = "worker" | "chief";

export interface AgentSummary {
  id: string;
  name: string;
  role: AgentRole;
  /** The instruction the agent was started with. */
  task: string;
  /** Linked kanban issue number, if any. */
  issue: number | null;
  state: AgentState;
  model: string;
  /** The worker's git branch and worktree path; null for the chief of staff. */
  branch: string | null;
  worktree: string | null;
  createdAt: string;
  updatedAt: string;
  /** ISO time of the newest journal entry, or null before the first. */
  lastActivity: string | null;
  turns: number;
  /** Client-side cost estimate from the session, when reported. */
  costUsd: number | null;
  sessionId: string | null;
  /** A fixed, path-free message when the agent failed. */
  error: string | null;
  pendingApprovals: number;
  /** Sequence number of the newest journal entry; the browser fetches entries after the one it has. */
  lastSeq: number;
}

export type AgentEventBody =
  | { kind: "user"; text: string }
  | { kind: "text"; text: string }
  | { kind: "tool-use"; tool: string; toolUseId: string; input: Record<string, unknown> }
  | { kind: "tool-result"; toolUseId: string; output: string; isError: boolean }
  | { kind: "state"; state: AgentState; detail?: string }
  | { kind: "approval"; approvalId: string; tool: string; outcome: "requested" | "allow" | "deny" | "cancelled" }
  | { kind: "result"; ok: boolean; text: string; turns: number; costUsd: number | null }
  | { kind: "error"; message: string };

export type AgentEvent = AgentEventBody & { seq: number; at: string };

export interface AgentsResponse {
  agents: AgentSummary[];
  /** Workers allowed at once. */
  max: number;
  driver: "sdk" | "fake";
}

/** A project skill under .claude/skills, with when it last ran in a Claude Code session. */
export interface SkillSummary {
  name: string;
  description: string;
  /** ISO time of the newest transcript entry that invoked the skill, or null if it never ran. */
  lastRun: string | null;
  /** Which cartoon to show, from `dashboard-art:` in SKILL.md; null gives the generic one. */
  art: string | null;
  /** False when the SKILL.md header could not be parsed; the skill is listed but may not be invocable. */
  readable: boolean;
}

export interface SkillsResponse {
  skills: SkillSummary[];
}

export interface AgentEventsResponse {
  events: AgentEvent[];
  /** Newest sequence number in the journal, so the browser knows when it has everything. */
  lastSeq: number;
}

/** A tool call an agent wants to make that needs a human decision. */
export interface Approval {
  id: string;
  agentId: string;
  agentName: string;
  tool: string;
  /** The exact input the agent supplied (a Bash command, a file path and content, an MCP tool's arguments). */
  input: Record<string, unknown>;
  /** One line describing the request, derived from `input`, for lists and screen readers. */
  summary: string;
  /** Why the permission system asked, when it said. */
  reason: string | null;
  requestedAt: string;
}

export type ApprovalDecision = "allow" | "deny";

export interface ApprovalsResponse {
  approvals: Approval[];
}

// ---- Chief of staff (milestone 2b) ----

export interface ChiefSummary {
  /** `idle` also covers "no session yet": the session starts with the first message. */
  state: AgentState;
  model: string;
  turns: number;
  costUsd: number | null;
  sessionId: string | null;
  error: string | null;
  pendingApprovals: number;
  lastSeq: number;
}

// ---- Terminal (milestone 3) ----

export interface TerminalStatus {
  enabled: boolean;
  /** Streams (browser connections) allowed at once. */
  max: number;
  open: number;
}
