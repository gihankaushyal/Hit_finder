import { forwardRef, type MouseEvent } from "react";
import { Crosshair, GitBranch, Robot } from "@phosphor-icons/react";
import { Badge, type BadgeTone } from "./Badge";
import { CopyButton } from "./CopyButton";
import { ThemeToggle } from "./ThemeToggle";
import { ciWord, testWord, type StatusReading } from "../lib/status";
import { tabHref, useSearch } from "../lib/useQueryState";
import { AGENTS_REFRESH_MS, CI_REFRESH_MS, REPO_REFRESH_MS, useResource } from "../lib/useResource";
import type { ApprovalsResponse, CiRun, TestRunState } from "../../../shared/types";
import "./DashboardNav.css";

const REPO_NAME = "Hit_finder";

export type TabId = "overview" | "kanban" | "runs";
export const DEFAULT_TAB: TabId = "overview";
export const TABS: { id: TabId; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "kanban", label: "Kanban Board" },
  { id: "runs", label: "Runs & Checkpoints" },
];
export const asTab = (v: string): TabId => (TABS.some((t) => t.id === v) ? (v as TabId) : DEFAULT_TAB);

interface RepoInfo {
  branch: string;
  commit: string;
  /** Older servers may not send it; the short hash is copied then. */
  commitFull?: string;
}

const TONE: Record<StatusReading["kind"], BadgeTone> = { ok: "ok", fail: "fail", running: "info", idle: "neutral", attention: "warn" };

/** A coloured dot is allowed here: it carries real state, and the word beside it says the same thing. */
function StatusBadge({ label, reading }: { label: string; reading: StatusReading }) {
  return (
    <Badge tone={TONE[reading.kind]} icon={<span className="nav-dot" data-kind={reading.kind} aria-hidden="true" />}>
      {label} {reading.text}
    </Badge>
  );
}

interface NavProps {
  tab: TabId;
  onTab: (tab: TabId) => void;
  agentsOpen: boolean;
  onToggleAgents: () => void;
}

export const DashboardNav = forwardRef<HTMLButtonElement, NavProps>(function DashboardNav({ tab, onTab, agentsOpen, onToggleAgents }, agentsButton) {
  const repo = useResource<RepoInfo>("repo", "/api/repo", undefined, REPO_REFRESH_MS);
  const ci = useResource<{ runs: CiRun[] }>("ci", "/api/ci", (d) => d.runs.length === 0, CI_REFRESH_MS);
  const tests = useResource<TestRunState>("tests", "/api/tests");
  const approvals = useResource<ApprovalsResponse>("approvals", "/api/approvals", (d) => d.approvals.length === 0, AGENTS_REFRESH_MS);
  const waiting = approvals.data?.approvals.length ?? 0;
  const ciReading: StatusReading = ci.error ? { kind: "attention", text: "Unknown" } : ciWord(ci.data?.runs);
  const testReading: StatusReading = tests.error ? { kind: "attention", text: "Unknown" } : testWord(tests.data);
  const search = useSearch();
  const go = (id: TabId) => (e: MouseEvent<HTMLAnchorElement>) => {
    // Leave Cmd/Ctrl/middle-click alone so a tab can open in a new window.
    if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    e.preventDefault();
    onTab(id);
  };
  return (
    <header className="nav">
      <div className="nav__bar">
        <h1 className="nav__brand">
          <Crosshair size={18} weight="bold" aria-hidden="true" />
          <span translate="no">{REPO_NAME}</span>
        </h1>
        <span className="nav__branch mono" translate="no">
          <GitBranch size={13} aria-hidden="true" />
          <span className="sr-only">Current branch: </span>
          <span>{repo.data?.branch ?? "unknown"}</span>
        </span>
        {repo.data?.commit && (
          <span className="nav__commit">
            <span className="mono" translate="no">{repo.data.commit}</span>
            <CopyButton text={repo.data.commitFull ?? repo.data.commit} label="Copy commit hash" />
          </span>
        )}
        <span className="nav__badges">
          <StatusBadge label="CI" reading={ciReading} />
          <StatusBadge label="Tests" reading={testReading} />
        </span>
        <span className="nav__spacer" />
        <button
          ref={agentsButton} type="button" className="btn btn--sm nav__agents" aria-haspopup="dialog" aria-expanded={agentsOpen}
          aria-controls="agents-drawer" aria-label={waiting > 0 ? `Agents, ${waiting} waiting for approval` : "Agents"} onClick={onToggleAgents}
        >
          <Robot size={15} aria-hidden="true" />
          <span>Agents</span>
          {waiting > 0 && (
            <span className="nav__count" aria-hidden="true">{waiting}</span>
          )}
        </button>
        <ThemeToggle />
      </div>
      <nav className="nav__tabs" aria-label="Dashboard sections">
        <ul>
          {TABS.map((t) => (
            <li key={t.id}>
              <a href={tabHref(search, t.id, DEFAULT_TAB)} aria-current={tab === t.id ? "page" : undefined} onClick={go(t.id)}>
                {t.label}
              </a>
            </li>
          ))}
        </ul>
      </nav>
    </header>
  );
});
