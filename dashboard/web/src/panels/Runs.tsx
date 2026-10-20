import { useId, useMemo, useState, type KeyboardEvent } from "react";
import { CaretDown, CaretRight, CaretUp, CaretUpDown } from "@phosphor-icons/react";
import { PanelFrame } from "../components/PanelFrame";
import { StatusWord, type StatusKind } from "../components/StatusWord";
import { StaleNotice, absoluteTime, panelState } from "../components/bits";
import { SORT_LABEL, defaultDirection, describeSort, sortRuns, type SortDir, type SortKey } from "../lib/runSort";
import { relTime } from "../lib/time";
import { RUNS_REFRESH_MS, useResource } from "../lib/useResource";
import type { MetricBlock, RunKind, RunResult, RunRow, RunStatus, RunsResponse } from "../../../shared/types";

const ICON_SIZE = 14;
/** Mirrors MAX_RUN_DIRS in server/lib/runScan.ts; shown in the cut-off notice. */
const SCAN_LIMIT_TEXT = 500;
const NA = "n/a";

const KIND_LABEL: Record<RunKind, string> = {
  track1: "Track 1",
  "ssl-pretrain": "SSL pretrain",
  "ssl-finetune": "SSL fine-tune",
  "ssl-probe": "SSL probe",
  legacy: "Legacy",
};
const STATUS: Record<RunStatus, { kind: StatusKind; text: string }> = {
  complete: { kind: "ok", text: "Complete" },
  provisional: { kind: "attention", text: "Provisional" },
  unknown: { kind: "idle", text: "Unknown" },
  "no-results": { kind: "idle", text: "No results" },
  pretrain: { kind: "idle", text: "Pretrain checkpoints" },
};

const fmt = (v: number | null | undefined) => (typeof v === "number" ? v.toFixed(3) : NA);
const metric = (b: MetricBlock | null | undefined, key: string) => fmt(b?.[key]);

function epochsText(r: RunRow): string {
  if (!r.epochs) return NA;
  if (r.status === "pretrain") return r.epochs.last === null ? NA : `epoch ${r.epochs.last}`;
  if (r.epochs.last !== null && r.epochs.configured !== null) return `${r.epochs.last} / ${r.epochs.configured}`;
  return r.epochs.last === null ? NA : String(r.epochs.last);
}

const isEmpty = (d: RunsResponse) => d.runs.length === 0;

function MetricTable({ title, block }: { title: string; block: MetricBlock | null }) {
  if (!block) return <p className="dim">{title}: {NA}</p>;
  return (
    <table className="runs-metrics">
      <caption>{title}</caption>
      <tbody>
        {Object.entries(block).map(([k, v]) => (
          <tr key={k}>
            <th scope="row" className="mono">{k}</th>
            <td className="mono">{fmt(v)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function ResultDetails({ result }: { result: RunResult | null }) {
  if (!result) return <p className="dim">No results.json for this run yet.</p>;
  const inf = result.inference;
  return (
    <div className="runs-detail">
      <dl className="runs-facts">
        <dt>Held-out detector</dt>
        <dd className="mono">{result.testDetector ?? NA}</dd>
        <dt>Threshold</dt>
        <dd className="mono">{fmt(result.inferenceThreshold)}</dd>
        {result.track !== null && (
          <>
            <dt>Track</dt>
            <dd className="mono">{result.track}{result.probe ? " (probe)" : ""}</dd>
          </>
        )}
        {result.pretrainEpoch !== null && (
          <>
            <dt>Pretrain epoch</dt>
            <dd className="mono">{result.pretrainEpoch}</dd>
          </>
        )}
        {inf && (
          <>
            <dt>Aggregation</dt>
            <dd className="mono">{inf.aggregation ?? NA}</dd>
            <dt>Checkpoint epoch</dt>
            <dd className="mono">{inf.checkpointEpoch ?? NA}</dd>
            <dt>Training check</dt>
            <dd className="mono">{inf.trainingStatus ?? NA}{inf.wandbState ? `, W&B ${inf.wandbState}` : ""}</dd>
            {inf.evaluatedAt && (
              <>
                <dt>Evaluated</dt>
                <dd><time dateTime={inf.evaluatedAt} title={absoluteTime(inf.evaluatedAt)} className="mono">{inf.evaluatedAt}</time></dd>
              </>
            )}
          </>
        )}
      </dl>
      <div className="runs-metric-pair">
        <MetricTable title="Cross-detector" block={result.cross} />
        <MetricTable title="In-domain" block={result.inDomain} />
      </div>
      {inf && inf.warnings.length > 0 && (
        <ul className="runs-warnings" aria-label="Warnings">
          {inf.warnings.map((w, i) => <li key={i}>{w}</li>)}
        </ul>
      )}
    </div>
  );
}

interface SortState { key: SortKey; dir: SortDir }

function SortHeader({ k, sort, onSort, numeric }: { k: SortKey; sort: SortState; onSort: (k: SortKey) => void; numeric?: boolean }) {
  const active = sort.key === k;
  const Icon = !active ? CaretUpDown : sort.dir === "asc" ? CaretUp : CaretDown;
  return (
    <th scope="col" className={numeric ? "num" : undefined} aria-sort={!active ? "none" : sort.dir === "asc" ? "ascending" : "descending"}>
      <button type="button" className="btn btn--bare runs-sort" onClick={() => onSort(k)}>
        <span>{SORT_LABEL[k]}</span>
        <Icon size={12} weight={active ? "bold" : "regular"} aria-hidden="true" />
      </button>
    </th>
  );
}

type Tab = "result" | "reeval";

function Detail({ run }: { run: RunRow }) {
  const ids = useId();
  const tabs: Tab[] = run.reevaluation ? ["result", "reeval"] : ["result"];
  const [chosen, setTab] = useState<Tab>("result");
  // A refresh can drop the re-evaluation while its tab is selected.
  const tab: Tab = tabs.includes(chosen) ? chosen : "result";
  const onKey = (e: KeyboardEvent<HTMLButtonElement>) => {
    if (tabs.length < 2 || (e.key !== "ArrowRight" && e.key !== "ArrowLeft")) return;
    e.preventDefault();
    const next: Tab = tab === "result" ? "reeval" : "result";
    setTab(next);
    document.getElementById(`${ids}-${next}`)?.focus();
  };
  return (
    <div>
      {run.error && <p role="alert" className="runs-error">{run.error}</p>}
      <div role="tablist" aria-label={`Views of ${run.name}`} className="tabs">
        {tabs.map((t) => (
          <button
            key={t} type="button" role="tab" id={`${ids}-${t}`} aria-selected={tab === t} aria-controls={`${ids}-panel`}
            tabIndex={tab === t ? 0 : -1} className="btn tabs__tab" onClick={() => setTab(t)} onKeyDown={onKey}
          >
            {t === "result" ? "Result" : "Re-evaluation"}
          </button>
        ))}
      </div>
      <div role="tabpanel" id={`${ids}-panel`} aria-labelledby={`${ids}-${tab}`}>
        {tab === "reeval" && <p className="dim">Re-evaluation, not the fold's result. It may use different aggregation settings.</p>}
        <ResultDetails result={tab === "result" ? run.result : run.reevaluation} />
      </div>
    </div>
  );
}

function Row({ run, open, onToggle }: { run: RunRow; open: boolean; onToggle: () => void }) {
  const st = STATUS[run.status];
  const Caret = open ? CaretDown : CaretRight;
  return (
    <>
      <tr>
        <td className="runs__name">
          <button type="button" className="btn btn--bare" aria-expanded={open} onClick={onToggle}>
            <Caret size={ICON_SIZE} aria-hidden="true" />
            <span className="mono">{run.name}</span>
          </button>
          <span className="runs__chips">
            <span className="chip">{KIND_LABEL[run.kind]}</span>
            {run.kind === "legacy" && <span className="chip">Legacy name</span>}
            {run.archivedV1 && <span className="chip">Archived v1</span>}
            {run.files.resultsInferenceJson && <span className="chip">Re-evaluated</span>}
          </span>
        </td>
        <td><StatusWord kind={st.kind}>{st.text}</StatusWord></td>
        <td className="mono">{run.result?.testDetector ?? NA}</td>
        <td className="mono num">{metric(run.result?.cross, "ap")}</td>
        <td className="mono num">{metric(run.result?.cross, "f1")}</td>
        <td className="mono num">{metric(run.result?.inDomain, "ap")}</td>
        <td className="mono num">{epochsText(run)}</td>
        <td className="mono num">
          {run.files.newestMtime ? <time dateTime={run.files.newestMtime} title={absoluteTime(run.files.newestMtime)}>{relTime(run.files.newestMtime)}</time> : NA}
        </td>
      </tr>
      {open && (
        <tr>
          <td colSpan={8} className="runs__detail"><Detail run={run} /></td>
        </tr>
      )}
    </>
  );
}

export function Runs() {
  const res = useResource<RunsResponse>("runs", "/api/runs", isEmpty, RUNS_REFRESH_MS);
  const [kind, setKind] = useState<RunKind | "all">("all");
  const [showOld, setShowOld] = useState(true);
  const [openName, setOpenName] = useState<string | null>(null);
  const [sort, setSort] = useState<SortState>({ key: "name", dir: "asc" });
  const [announcement, setAnnouncement] = useState("");
  const kindId = useId();
  const data = res.data;
  const shown = useMemo(
    () => sortRuns((data?.runs ?? []).filter((r) => (kind === "all" || r.kind === kind) && (showOld || (r.kind !== "legacy" && !r.archivedV1))), sort.key, sort.dir),
    [data, kind, showOld, sort],
  );
  const onSort = (k: SortKey) => {
    const next: SortState = sort.key === k ? { key: k, dir: sort.dir === "asc" ? "desc" : "asc" } : { key: k, dir: defaultDirection(k) };
    setSort(next);
    setAnnouncement(describeSort(next.key, next.dir));
  };
  const actions =
    data && data.runs.length > 0 ? (
      <>
        <span className="runs-filter">
          <label htmlFor={kindId}>Kind</label>
          <select id={kindId} value={kind} onChange={(e) => setKind(e.target.value as RunKind | "all")}>
            <option value="all">All kinds</option>
            {(Object.keys(KIND_LABEL) as RunKind[]).map((k) => <option key={k} value={k}>{KIND_LABEL[k]}</option>)}
          </select>
        </span>
        <label className="runs-filter">
          <input type="checkbox" checked={showOld} onChange={(e) => setShowOld(e.target.checked)} />
          Show archived and legacy runs
        </label>
      </>
    ) : undefined;
  return (
    <PanelFrame title="Runs" count={data ? shown.length : undefined} actions={actions} state={panelState(res)} error={res.error ?? undefined} onRetry={res.reload}>
      <StaleNotice res={res} label="runs" />
      <div role="status" className="sr-only">{announcement}</div>
      {data && res.empty && !data.error && (
        <p className="dim">
          {data.checkpointsDirExists
            ? "No runs found under checkpoints/. Runs appear here once training writes best.pt, last.pt or results.json."
            : "No checkpoints directory on this machine. The Runs panel reads checkpoints/ next to the repository, so it shows runs only when the dashboard runs on the same filesystem."}
        </p>
      )}
      {data?.error && <p role="alert" className="runs-error">{data.error}</p>}
      {data?.truncated && <p className="dim">Only the first {SCAN_LIMIT_TEXT} run directories are listed.</p>}
      {data && !res.empty && shown.length === 0 && <p className="dim">No runs match the current filters.</p>}
      {data && !res.empty && shown.length > 0 && (
        <div className="runs-wrap">
          <table className="runs" aria-label="Training runs">
            <thead>
              <tr>
                <SortHeader k="name" sort={sort} onSort={onSort} />
                <SortHeader k="status" sort={sort} onSort={onSort} />
                <SortHeader k="detector" sort={sort} onSort={onSort} />
                <SortHeader k="crossAp" sort={sort} onSort={onSort} numeric />
                <SortHeader k="crossF1" sort={sort} onSort={onSort} numeric />
                <SortHeader k="inDomainAp" sort={sort} onSort={onSort} numeric />
                <SortHeader k="epochs" sort={sort} onSort={onSort} numeric />
                <SortHeader k="updated" sort={sort} onSort={onSort} numeric />
              </tr>
            </thead>
            <tbody>
              {shown.map((r) => (
                <Row key={r.name} run={r} open={openName === r.name} onToggle={() => setOpenName(openName === r.name ? null : r.name)} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </PanelFrame>
  );
}
