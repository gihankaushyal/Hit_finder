import { useState } from "react";
import { CheckCircle, FileText, GitMerge, GitPullRequest, Hourglass, ProhibitInset, XCircle } from "@phosphor-icons/react";
import { Badge } from "../components/Badge";
import { DiffChips } from "../components/DiffChips";
import { PanelFrame } from "../components/PanelFrame";
import { ExternalLink, StaleNotice, absoluteTime, panelState } from "../components/bits";
import { renderInline } from "../lib/inlineMarkdown";
import { relTime } from "../lib/time";
import { PRS_REFRESH_MS, useResource } from "../lib/useResource";
import type { PrChecks, PrDetail, PrSummary, PrsResponse } from "../../../shared/types";
import "./PrCard.css";

/** Longer file lists are cut to this many rows until the user asks for all of them. */
export const FILES_COLLAPSED_COUNT = 8;
const ICON_SIZE = 13;
const FILE_ICON_SIZE = 14;

const isEmpty = (d: PrsResponse) => d.latest === null && d.open.length === 0 && d.recent.length === 0;

function Branches({ head, base }: { head: string; base: string }) {
  return (
    <span className="pr-branches">
      <span className="pr-branch mono">{head}</span> into <span className="pr-branch mono">{base}</span>
    </span>
  );
}

function When({ iso }: { iso: string }) {
  return (
    <time dateTime={iso} title={absoluteTime(iso)}>
      {relTime(iso)}
    </time>
  );
}

function StateBadge({ state }: { state: PrSummary["state"] }) {
  if (state === "MERGED") return <Badge tone="merged" icon={<GitMerge size={ICON_SIZE} weight="bold" aria-hidden="true" />}>Merged</Badge>;
  if (state === "OPEN") return <Badge tone="open" icon={<GitPullRequest size={ICON_SIZE} weight="bold" aria-hidden="true" />}>Open</Badge>;
  return <Badge tone="fail" icon={<ProhibitInset size={ICON_SIZE} weight="bold" aria-hidden="true" />}>Closed</Badge>;
}

function Checks({ checks }: { checks: PrChecks | undefined }) {
  if (!checks || checks.state === "none") return <span className="dim">No checks reported</span>;
  const total = checks.passing + checks.failing + checks.pending;
  if (checks.state === "failing") {
    return <Badge tone="fail" icon={<XCircle size={ICON_SIZE} weight="bold" aria-hidden="true" />}>Checks failing ({checks.failing} of {total})</Badge>;
  }
  if (checks.state === "pending") {
    return <Badge tone="info" icon={<Hourglass size={ICON_SIZE} weight="bold" aria-hidden="true" />}>Checks running ({checks.pending} of {total})</Badge>;
  }
  return <Badge tone="ok" icon={<CheckCircle size={ICON_SIZE} weight="bold" aria-hidden="true" />}>Checks passed ({checks.passing})</Badge>;
}

function Files({ files }: { files: PrDetail["files"] }) {
  const [all, setAll] = useState(false);
  const shown = all ? files : files.slice(0, FILES_COLLAPSED_COUNT);
  return (
    <>
      <ul className="plain pr-files" aria-label="Touched files">
        {shown.map((f) => (
          <li key={f.path} className="pr-file">
            <FileText size={FILE_ICON_SIZE} aria-hidden="true" className="pr-file__icon" />
            <span className="pr-file__path mono" translate="no">{f.path}</span>
            <DiffChips additions={f.additions} deletions={f.deletions} />
          </li>
        ))}
      </ul>
      {files.length > FILES_COLLAPSED_COUNT && (
        <button type="button" className="btn btn--quiet" aria-expanded={all} onClick={() => setAll(!all)}>
          {all ? "Show fewer files" : `Show all ${files.length} files`}
        </button>
      )}
    </>
  );
}

function TestPlan({ items }: { items: PrDetail["testPlan"] }) {
  return (
    <ul className="plain pr-plan" aria-label="Test plan">
      {items.map((t, i) => (
        <li key={i}>
          <label className="pr-check" data-checked={t.checked}>
            <input type="checkbox" name={`plan-${i}`} checked={t.checked} disabled readOnly />
            <span className="pr-check__text">{t.checked ? <s>{renderInline(t.text)}</s> : renderInline(t.text)}</span>
          </label>
        </li>
      ))}
    </ul>
  );
}

export function PrCard() {
  const prs = useResource<PrsResponse>("prs", "/api/prs", isEmpty, PRS_REFRESH_MS);
  const state = panelState(prs);
  const data = prs.data;
  const latest = data?.latest ?? null;
  return (
    <PanelFrame title="Latest pull request" state={state} error={prs.error ?? undefined} onRetry={prs.reload}>
      <StaleNotice res={prs} label="pull requests" />
      {data && data.open.length > 0 && (
        <ul className="rows pr-open" aria-label="Open pull requests">
          {data.open.map((p) => (
            <li key={p.number} className="row row--wrap">
              <Badge tone="open" icon={<GitPullRequest size={ICON_SIZE} weight="bold" aria-hidden="true" />}>Open</Badge>
              <ExternalLink href={p.url}>{p.title}</ExternalLink>
              <span className="mono faint">#{p.number}</span>
              <span className="dim"><Branches head={p.headRefName} base={p.baseRefName} /></span>
              <Checks checks={p.checks} />
            </li>
          ))}
        </ul>
      )}
      {latest ? (
        <article className="pr">
          <h3 className="pr__title">
            <span className="mono faint">#{latest.number}</span> <ExternalLink href={latest.url}>{latest.title}</ExternalLink>
          </h3>
          <div className="pr__badges">
            <StateBadge state={latest.state} />
            <DiffChips additions={latest.additions} deletions={latest.deletions} />
          </div>
          <p className="meta">
            <Branches head={latest.headRefName} base={latest.baseRefName} />
            {latest.mergedAt && <span>merged <When iso={latest.mergedAt} /></span>}
            <span><span className="mono">{latest.changedFiles}</span> {latest.changedFiles === 1 ? "file" : "files"}</span>
          </p>
          {latest.summary.length > 0 && (
            <>
              <h4 className="sub">What it did</h4>
              <ul className="bullets" aria-label="What it did">
                {latest.summary.map((s, i) => <li key={i}>{renderInline(s)}</li>)}
              </ul>
            </>
          )}
          {latest.testPlan.length > 0 && (
            <>
              <h4 className="sub">Test plan</h4>
              <TestPlan items={latest.testPlan} />
            </>
          )}
          {latest.files.length > 0 && (
            <>
              <h4 className="sub">Files ({latest.files.length})</h4>
              <Files files={latest.files} />
            </>
          )}
        </article>
      ) : (
        <p className="dim">No merged pull requests yet. They appear here once a pull request is merged on GitHub.</p>
      )}
      {data && data.recent.length > 0 && (
        <>
          <h3 className="sub">Earlier</h3>
          <ul className="rows" aria-label="Earlier pull requests">
            {data.recent.map((p) => (
              <li key={p.number} className="row row--wrap">
                <span className="mono faint">#{p.number}</span>
                <ExternalLink href={p.url}>{p.title}</ExternalLink>
                {p.mergedAt && <span className="dim"><When iso={p.mergedAt} /></span>}
              </li>
            ))}
          </ul>
        </>
      )}
    </PanelFrame>
  );
}
