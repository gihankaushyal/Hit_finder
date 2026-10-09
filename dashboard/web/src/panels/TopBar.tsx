import { GitBranch } from "@phosphor-icons/react";
import { StatusWord, type StatusKind } from "../components/StatusWord";
import { ThemeToggle } from "../components/ThemeToggle";
import { CI_REFRESH_MS, REPO_REFRESH_MS, useResource } from "../lib/useResource";
import type { CiRun, TestRunState } from "../../../shared/types";

const REPO_NAME = "Hit_finder";

interface RepoInfo {
  branch: string;
  commit: string;
}

function ciWord(runs: CiRun[] | undefined): { kind: StatusKind; text: string } {
  const run = runs?.[0];
  if (!run) return { kind: "idle", text: "No runs" };
  if (run.status !== "completed") return { kind: "running", text: "Running" };
  if (run.conclusion === "success") return { kind: "ok", text: "Passing" };
  if (run.conclusion === "cancelled" || run.conclusion === "skipped") return { kind: "idle", text: "Cancelled" };
  return { kind: "fail", text: "Failing" };
}

function testWord(t: TestRunState | null): { kind: StatusKind; text: string } {
  if (!t || t.status === "idle") return { kind: "idle", text: "Not run" };
  if (t.status === "running") return { kind: "running", text: "Running" };
  if (t.status === "error") return { kind: "fail", text: "Error" };
  const bad = (t.summary?.failed ?? 0) + (t.summary?.errors ?? 0);
  return bad > 0 ? { kind: "fail", text: "Failing" } : { kind: "ok", text: "Passing" };
}

export function TopBar() {
  const repo = useResource<RepoInfo>("repo", "/api/repo", undefined, REPO_REFRESH_MS);
  const ci = useResource<{ runs: CiRun[] }>("ci", "/api/ci", (d) => d.runs.length === 0, CI_REFRESH_MS);
  const tests = useResource<TestRunState>("tests", "/api/tests");
  const ciState = ci.error ? { kind: "attention" as const, text: "Unknown" } : ciWord(ci.data?.runs);
  const testState = tests.error ? { kind: "attention" as const, text: "Unknown" } : testWord(tests.data);
  return (
    <header className="topbar">
      <h1 className="topbar__title">{REPO_NAME}</h1>
      <span className="topbar__branch mono">
        <GitBranch size={14} aria-hidden="true" />
        <span aria-label="Current branch">{repo.data?.branch ?? "unknown"}</span>
        {repo.data?.commit && <span className="dim"> {repo.data.commit}</span>}
      </span>
      <span className="topbar__spacer" />
      <span className="topbar__item">
        <span className="dim">CI</span> <StatusWord kind={ciState.kind}>{ciState.text}</StatusWord>
      </span>
      <span className="topbar__item">
        <span className="dim">Tests</span> <StatusWord kind={testState.kind}>{testState.text}</StatusWord>
      </span>
      <ThemeToggle />
    </header>
  );
}
