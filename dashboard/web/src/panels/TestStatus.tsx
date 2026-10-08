import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { Play } from "@phosphor-icons/react";
import { PanelFrame } from "../components/PanelFrame";
import { StatusWord, type StatusKind } from "../components/StatusWord";
import { ExternalLink, SectionState, StaleNotice } from "../components/bits";
import { ApiError, apiSend } from "../lib/api";
import { subscribe } from "../lib/events";
import { duration, relTime } from "../lib/time";
import { useResource } from "../lib/useResource";
import type { CiRun, TestRunState } from "../../../shared/types";

const EARLIER_CI_RUNS = 5;
const MAX_LOG_LINES = 500;
/** Within this many pixels of the bottom counts as "at the tail". */
const TAIL_SLACK_PX = 6;
const SHORT_SHA = 7;
const ALREADY_RUNNING = "A test run is already in progress";

function ciWord(run: CiRun): { kind: StatusKind; text: string } {
  if (run.status !== "completed") return { kind: "running", text: "Running" };
  if (run.conclusion === "success") return { kind: "ok", text: "Passed" };
  if (run.conclusion === "cancelled" || run.conclusion === "skipped") return { kind: "idle", text: run.conclusion === "skipped" ? "Skipped" : "Cancelled" };
  return { kind: "fail", text: "Failed" };
}

function localWord(t: TestRunState): { kind: StatusKind; text: string } {
  if (t.status === "idle") return { kind: "idle", text: "Not run" };
  if (t.status === "running") return { kind: "running", text: "Running" };
  if (t.status === "error") return { kind: "fail", text: "Error" };
  return (t.summary?.failed ?? 0) + (t.summary?.errors ?? 0) > 0 ? { kind: "fail", text: "Failed" } : { kind: "ok", text: "Passed" };
}

function GithubActions() {
  const ci = useResource<{ runs: CiRun[] }>("ci", "/api/ci", (d) => d.runs.length === 0);
  const runs = ci.data?.runs ?? [];
  const [first, ...rest] = runs;
  return (
    <section className="subpanel" aria-labelledby="ts-gh">
      <h3 id="ts-gh" className="sub">GitHub Actions</h3>
      {ci.data === null ? (
        <SectionState res={ci} label="GitHub Actions" />
      ) : !first ? (
        <p className="dim">No CI runs found for this repository.</p>
      ) : (
        <>
          <StaleNotice res={ci} label="CI runs" />
          <p className="row row--wrap">
            <StatusWord kind={ciWord(first).kind}>{ciWord(first).text}</StatusWord>
            <ExternalLink href={first.url} className="mono">{first.branch}</ExternalLink>
            <span className="dim">{relTime(first.createdAt)}</span>
          </p>
          {rest.length > 0 && (
            <ul className="rows" aria-label="Earlier CI runs">
              {rest.slice(0, EARLIER_CI_RUNS).map((r) => (
                <li key={r.id} className="row row--wrap dim">
                  <span>{ciWord(r).text}</span>
                  <ExternalLink href={r.url} className="mono">{r.branch}</ExternalLink>
                  <span>{relTime(r.createdAt)}</span>
                </li>
              ))}
            </ul>
          )}
        </>
      )}
    </section>
  );
}

/** Output lines: what the server already has, plus lines streamed since. */
function useTestLines(tail: string[] | undefined): string[] {
  const [lines, setLines] = useState<string[]>([]);
  useEffect(() => {
    if (tail) setLines(tail.slice(-MAX_LOG_LINES));
  }, [tail]);
  useEffect(
    () =>
      subscribe({
        resource: "tests-line",
        reload: () => {},
        onLine: (line) => setLines((prev) => [...prev, line].slice(-MAX_LOG_LINES)),
      }),
    [],
  );
  return lines;
}

function Log({ lines }: { lines: string[] }) {
  const ref = useRef<HTMLPreElement>(null);
  const [following, setFollowing] = useState(true);
  const followRef = useRef(true);
  const toTail = () => {
    const el = ref.current;
    if (el) el.scrollTop = el.scrollHeight;
  };
  useLayoutEffect(() => {
    if (followRef.current) toTail();
  }, [lines]);
  const onScroll = () => {
    const el = ref.current;
    if (!el) return;
    const atTail = el.scrollHeight - el.scrollTop - el.clientHeight <= TAIL_SLACK_PX;
    followRef.current = atTail;
    setFollowing(atTail);
  };
  return (
    <div className="log">
      {/* Focusable so keyboard users can scroll it. */}
      <pre ref={ref} className="log__body mono" aria-live="off" aria-label="Test output" tabIndex={0} onScroll={onScroll}>
        {lines.join("\n")}
      </pre>
      {!following && (
        <button type="button" className="btn btn--quiet log__follow" onClick={() => { followRef.current = true; setFollowing(true); toTail(); }}>
          Follow output
        </button>
      )}
    </div>
  );
}

function LocalRun() {
  const tests = useResource<TestRunState>("tests", "/api/tests", () => false);
  const t = tests.data;
  const lines = useTestLines(t?.tail);
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);
  const [announcement, setAnnouncement] = useState("");
  const prevStatus = useRef<TestRunState["status"] | null>(null);

  useEffect(() => {
    const now = t?.status ?? null;
    if (prevStatus.current === "running" && now === "done" && t?.summary) {
      setAnnouncement(`Local tests finished: ${t.summary.passed} passed, ${t.summary.failed + t.summary.errors} failed`);
    } else if (prevStatus.current === "running" && now === "error") {
      setAnnouncement("Local test run failed to complete");
    } else if (now === "running") {
      setAnnouncement("");
    }
    prevStatus.current = now;
  }, [t]);

  const run = async () => {
    setStarting(true);
    setStartError(null);
    try {
      await apiSend("POST", "/api/tests/run");
    } catch (err) {
      setStartError(err instanceof ApiError && err.status === 409 ? ALREADY_RUNNING : err instanceof ApiError ? err.message : "Could not start the run.");
    } finally {
      setStarting(false);
      tests.reload();
    }
  };

  const running = t?.status === "running";
  const word = t ? localWord(t) : null;
  const s = t?.summary;
  return (
    <section className="subpanel" aria-labelledby="ts-local">
      <h3 id="ts-local" className="sub">Local run</h3>
      <div role="status" aria-live="polite" className="sr-only">{announcement}</div>
      {t === null ? (
        <SectionState res={tests} label="Local run" />
      ) : (
        <>
          <StaleNotice res={tests} label="local run" />
          <div className="row row--wrap">
            {word && <StatusWord kind={word.kind}>{word.text}</StatusWord>}
            <button type="button" className="btn" disabled={running || starting} onClick={run}>
              <Play size={14} aria-hidden="true" />
              <span>{running ? "Running" : "Run tests"}</span>
            </button>
          </div>
          {startError && <p className="field__error" role="alert">{startError}</p>}
          {t.status === "idle" && <p className="dim">No local run yet. Run tests to start one.</p>}
          {t.status === "error" && t.error && <p className="field__error">{t.error}</p>}
          {s && (
            <>
              <p className="meta">
                <span><span className="mono">{s.passed}</span> passed</span>
                <span><span className="mono">{s.failed + s.errors}</span> failed</span>
                <span><span className="mono">{s.skipped}</span> skipped</span>
                <span className="mono">{duration(s.durationSec)}</span>
                {t.commit && <span className="dim">commit <span className="mono">{t.commit.slice(0, SHORT_SHA)}</span></span>}
              </p>
              {s.failures.length > 0 && (
                <ul className="plain" aria-label="Failing tests">
                  {s.failures.map((f) => (
                    <li key={f.name} className="failure">
                      <span className="mono">{f.name}</span>
                      <span className="dim">{f.message}</span>
                    </li>
                  ))}
                </ul>
              )}
            </>
          )}
          {(running || lines.length > 0) && <Log lines={lines} />}
        </>
      )}
    </section>
  );
}

export function TestStatus() {
  return (
    <PanelFrame title="Tests" state="ready">
      <div className="ts-grid">
        <GithubActions />
        <LocalRun />
      </div>
    </PanelFrame>
  );
}
