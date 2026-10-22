import type { StatusKind } from "../components/StatusWord";
import type { CiRun, TestRunState } from "../../../shared/types";

export interface StatusReading {
  kind: StatusKind;
  text: string;
}

/** One CI run as one word. Used by the nav badge and the Tests card, so both always agree. */
export function ciRunWord(run: CiRun): StatusReading {
  if (run.status !== "completed") return { kind: "running", text: "Running" };
  if (run.conclusion === "success") return { kind: "ok", text: "Passed" };
  if (run.conclusion === "cancelled") return { kind: "idle", text: "Cancelled" };
  if (run.conclusion === "skipped") return { kind: "idle", text: "Skipped" };
  return { kind: "fail", text: "Failed" };
}

/** The newest CI run, as one word for the nav. */
export function ciWord(runs: CiRun[] | undefined): StatusReading {
  const run = runs?.[0];
  return run ? ciRunWord(run) : { kind: "idle", text: "No runs" };
}

/** The local test run, as one word. */
export function testWord(t: TestRunState | null): StatusReading {
  if (!t || t.status === "idle") return { kind: "idle", text: "Not run" };
  if (t.status === "running") return { kind: "running", text: "Running" };
  if (t.status === "error") return { kind: "fail", text: "Error" };
  const bad = (t.summary?.failed ?? 0) + (t.summary?.errors ?? 0);
  return bad > 0 ? { kind: "fail", text: "Failed" } : { kind: "ok", text: "Passed" };
}
