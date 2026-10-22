import type { StatusKind } from "../components/StatusWord";
import type { CiRun, TestRunState } from "../../../shared/types";

export interface StatusReading {
  kind: StatusKind;
  text: string;
}

/** The newest CI run, as one word for the top bar. */
export function ciWord(runs: CiRun[] | undefined): StatusReading {
  const run = runs?.[0];
  if (!run) return { kind: "idle", text: "No runs" };
  if (run.status !== "completed") return { kind: "running", text: "Running" };
  if (run.conclusion === "success") return { kind: "ok", text: "Passing" };
  if (run.conclusion === "cancelled" || run.conclusion === "skipped") return { kind: "idle", text: "Cancelled" };
  return { kind: "fail", text: "Failing" };
}

/** The local test run, as one word for the top bar. */
export function testWord(t: TestRunState | null): StatusReading {
  if (!t || t.status === "idle") return { kind: "idle", text: "Not run" };
  if (t.status === "running") return { kind: "running", text: "Running" };
  if (t.status === "error") return { kind: "fail", text: "Error" };
  const bad = (t.summary?.failed ?? 0) + (t.summary?.errors ?? 0);
  return bad > 0 ? { kind: "fail", text: "Failing" } : { kind: "ok", text: "Passing" };
}
