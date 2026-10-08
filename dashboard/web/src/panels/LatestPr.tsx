import { useState } from "react";
import { CheckSquare, Square } from "@phosphor-icons/react";
import { PanelFrame } from "../components/PanelFrame";
import { StatusWord } from "../components/StatusWord";
import { ExternalLink, StaleNotice, absoluteTime, panelState } from "../components/bits";
import { relTime } from "../lib/time";
import { useResource } from "../lib/useResource";
import type { PrChecks, PrDetail, PrSummary, PrsResponse } from "../../../shared/types";

/** Longer file lists are cut to this many rows until the user asks for all of them. */
export const FILES_COLLAPSED_COUNT = 8;
const ICON_SIZE = 14;

const isEmpty = (d: PrsResponse) => d.latest === null && d.open.length === 0 && d.recent.length === 0;

function Branches({ head, base }: { head: string; base: string }) {
  return (
    <span>
      <span className="mono">{head}</span> into <span className="mono">{base}</span>
    </span>
  );
}

function Stats({ pr }: { pr: PrSummary }) {
  return (
    <span className="mono">
      <span className="add">+{pr.additions}</span> <span className="del">-{pr.deletions}</span>
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

function Checks({ checks }: { checks: PrChecks | undefined }) {
  if (!checks || checks.state === "none") return <span className="dim">No checks reported</span>;
  const total = checks.passing + checks.failing + checks.pending;
  if (checks.state === "failing") return <StatusWord kind="fail">Checks failing ({checks.failing} of {total})</StatusWord>;
  if (checks.state === "pending") return <StatusWord kind="running">Checks running ({checks.pending} of {total})</StatusWord>;
  return <StatusWord kind="ok">Checks passed ({checks.passing})</StatusWord>;
}

function Files({ files }: { files: PrDetail["files"] }) {
  const [all, setAll] = useState(false);
  const shown = all ? files : files.slice(0, FILES_COLLAPSED_COUNT);
  return (
    <>
      <ul className="plain mono" aria-label="Touched files">
        {shown.map((f) => (
          <li key={f.path} className="file-row">
            <span className="file-row__path">{f.path}</span>
            <span><span className="add">+{f.additions}</span> <span className="del">-{f.deletions}</span></span>
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

export function LatestPr() {
  const prs = useResource<PrsResponse>("prs", "/api/prs", isEmpty);
  const state = panelState(prs);
  const data = prs.data;
  const latest = data?.latest ?? null;
  return (
    <PanelFrame title="Latest pull request" state={state} error={prs.error ?? undefined} onRetry={prs.reload}>
      <StaleNotice res={prs} label="pull requests" />
      {data && data.open.length > 0 && (
        <ul className="rows" aria-label="Open pull requests">
          {data.open.map((p) => (
            <li key={p.number} className="row row--wrap">
              <StatusWord kind="attention">Open</StatusWord>
              <ExternalLink href={p.url}>{p.title}</ExternalLink>
              <span className="mono dim">#{p.number}</span>
              <span className="dim"><Branches head={p.headRefName} base={p.baseRefName} /></span>
              <Checks checks={p.checks} />
            </li>
          ))}
        </ul>
      )}
      {latest ? (
        <article className="pr">
          <h3 className="pr__title">
            <ExternalLink href={latest.url}>{latest.title}</ExternalLink> <span className="mono dim">#{latest.number}</span>
          </h3>
          <p className="meta">
            <Branches head={latest.headRefName} base={latest.baseRefName} />
            {latest.mergedAt && <span>merged <When iso={latest.mergedAt} /></span>}
            <Stats pr={latest} />
            <span><span className="mono">{latest.changedFiles}</span> {latest.changedFiles === 1 ? "file" : "files"}</span>
          </p>
          {latest.summary.length > 0 && (
            <>
              <h4 className="sub">What it did</h4>
              <ul className="bullets" aria-label="What it did">
                {latest.summary.map((s, i) => <li key={i}>{s}</li>)}
              </ul>
            </>
          )}
          {latest.testPlan.length > 0 && (
            <>
              <h4 className="sub">Test plan</h4>
              <ul className="plain" aria-label="Test plan">
                {latest.testPlan.map((t, i) => (
                  <li key={i} className="check-row">
                    {t.checked ? <CheckSquare size={ICON_SIZE} aria-hidden="true" /> : <Square size={ICON_SIZE} aria-hidden="true" />}
                    <span className="sr-only">{t.checked ? "Done: " : "Not done: "}</span>
                    {t.checked ? <s className="dim">{t.text}</s> : <span>{t.text}</span>}
                  </li>
                ))}
              </ul>
            </>
          )}
          {latest.files.length > 0 && (
            <>
              <h4 className="sub">Files <span className="mono dim">{latest.files.length}</span></h4>
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
                <span className="mono dim">#{p.number}</span>
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
