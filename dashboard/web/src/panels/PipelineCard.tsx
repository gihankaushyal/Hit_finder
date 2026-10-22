import { Fragment, memo, useEffect, useLayoutEffect, useRef, useState } from "react";
import { CheckCircle, CircleNotch, Play, Prohibit, XCircle } from "@phosphor-icons/react";
import { PanelFrame } from "../components/PanelFrame";
import { StatusWord } from "../components/StatusWord";
import { ExternalLink, SectionState, StaleNotice } from "../components/bits";
import { ApiError, apiSend } from "../lib/api";
import { ciRunWord, testWord } from "../lib/status";
import { subscribe } from "../lib/events";
import { duration, relTime } from "../lib/time";
import { CI_REFRESH_MS, useResource } from "../lib/useResource";
import type { CiRun, TestRunState } from "../../../shared/types";
import "./PipelineCard.css";

const EARLIER_CI_RUNS = 5;
const MAX_LOG_LINES = 500;
/** Within this many pixels of the bottom counts as "at the tail". */
const TAIL_SLACK_PX = 6;
const SHORT_SHA = 7;
const ALREADY_RUNNING = "A test run is already in progress";

const EVENT_LABEL: Record<string, string> = {
  push: "Push",
  pull_request: "Pull request",
  workflow_dispatch: "Manual run",
  schedule: "Scheduled run",
};
const RUN_ICON_SIZE = 16;
const eventLabel = (event: string) => EVENT_LABEL[event] ?? event.replace(/_/g, " ");

/** Icon per state; the word is for assistive tech, since colour and shape are not enough on their own. */
function RunIcon({ run }: { run: CiRun }) {
  const word = ciRunWord(run);
  const common = { size: RUN_ICON_SIZE, weight: "fill" as const, "aria-hidden": true };
  const icon =
    word.kind === "running" ? <CircleNotch {...common} weight="bold" className="spin" />
    : word.kind === "ok" ? <CheckCircle {...common} />
    : word.kind === "fail" ? <XCircle {...common} />
    : <Prohibit {...common} weight="bold" />;
  return (
    <span className="ci-run__icon" data-kind={word.kind}>
      {icon}
      <span className="sr-only">{word.text}</span>
    </span>
  );
}

function GithubActions() {
  const ci = useResource<{ runs: CiRun[] }>("ci", "/api/ci", (d) => d.runs.length === 0, CI_REFRESH_MS);
  const runs = (ci.data?.runs ?? []).slice(0, EARLIER_CI_RUNS + 1);
  return (
    <section className="subpanel" aria-labelledby="ts-gh">
      <h3 id="ts-gh" className="sub">GitHub Actions</h3>
      {ci.data === null ? (
        <SectionState res={ci} label="GitHub Actions" />
      ) : runs.length === 0 ? (
        <p className="dim">No CI runs found for this repository.</p>
      ) : (
        <>
          <StaleNotice res={ci} label="CI runs" />
          <ul className="plain ci-runs" aria-label="CI runs">
            {runs.map((r) => (
              <li key={r.id} className="ci-run">
                <RunIcon run={r} />
                <ExternalLink href={r.url} className="ci-run__title">{eventLabel(r.event)}</ExternalLink>
                <span className="ci-run__branch mono" translate="no">{r.branch}</span>
                <time className="ci-run__when dim" dateTime={r.createdAt}>{relTime(r.createdAt)}</time>
              </li>
            ))}
          </ul>
        </>
      )}
    </section>
  );
}

/** Output lines: what the server already has, plus lines streamed since. */
function useTestLines(tail: string[] | undefined): string[] {
  const [lines, setLines] = useState<string[]>([]);
  const pending = useRef<string[]>([]);
  const frame = useRef(0);
  useEffect(() => {
    pending.current = [];
    if (tail) setLines(tail.slice(-MAX_LOG_LINES));
  }, [tail]);
  useEffect(() => {
    // Lines can arrive faster than the screen refreshes; one render per frame is enough.
    const flush = () => {
      frame.current = 0;
      const batch = pending.current;
      pending.current = [];
      if (batch.length > 0) setLines((prev) => [...prev, ...batch].slice(-MAX_LOG_LINES));
    };
    const stop = subscribe({
      resource: "tests-line",
      reload: () => {},
      onLine: (line) => {
        pending.current.push(line);
        if (frame.current === 0) frame.current = requestAnimationFrame(flush);
      },
    });
    return () => {
      stop();
      cancelAnimationFrame(frame.current);
      frame.current = 0;
    };
  }, []);
  return lines;
}

const URL_SPLIT = /(https?:\/\/[^\s)]+)/;
const SUMMARY_LINE = /\b\d+ (?:passed|failed|skipped|errors?|warnings?)\b.*\bin [\d.]+s/;
const FAIL_LINE = /^(?:FAILED|ERROR)\b/;

/** One output line. Summary lines are yellow, URLs cyan, failures red; the text itself is never altered. */
const TermLine = memo(function TermLine({ text }: { text: string }) {
  if (SUMMARY_LINE.test(text)) return <span className="t-sum">{text}</span>;
  if (FAIL_LINE.test(text)) return <span className="t-fail">{text}</span>;
  const parts = text.split(URL_SPLIT);
  if (parts.length === 1) return <>{text}</>;
  return (
    <>
      {parts.map((p, i) => (i % 2 === 1 ? <span key={i} className="t-url">{p}</span> : p))}
    </>
  );
});

function Log({ lines, live }: { lines: string[]; live: boolean }) {
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
    <div className="log" data-live={live}>
      {/* Focusable so keyboard users can scroll it. The trailing shell prompt is drawn by CSS, so it is not part of the output text. */}
      <pre ref={ref} className="log__body mono" aria-live="off" aria-label="Test output" tabIndex={0} onScroll={onScroll}>
        {lines.map((line, i) => (
          <Fragment key={i}>
            <TermLine text={line} />
            {i < lines.length - 1 ? "\n" : ""}
          </Fragment>
        ))}
      </pre>
      {!following && (
        <button type="button" className="btn btn--sm log__follow" onClick={() => { followRef.current = true; setFollowing(true); toTail(); }}>
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
  const word = t ? testWord(t) : null;
  const s = t?.summary;
  return (
    <section className="subpanel" aria-labelledby="ts-local">
      <div className="local-head">
        <h3 id="ts-local" className="sub">Local run</h3>
        {word && <StatusWord kind={word.kind}>{word.text}</StatusWord>}
        {t !== null && (
          <button type="button" className="btn btn--sm local-head__run" disabled={running || starting} onClick={run}>
            <Play size={13} weight="fill" aria-hidden="true" />
            <span>{running ? "Running…" : "Run tests"}</span>
          </button>
        )}
      </div>
      <div role="status" aria-live="polite" className="sr-only">{announcement}</div>
      {t === null ? (
        <SectionState res={tests} label="Local run" />
      ) : (
        <>
          <StaleNotice res={tests} label="local run" />
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
          {(lines.length > 0 || running) && <Log lines={lines} live={running} />}
        </>
      )}
    </section>
  );
}

export function PipelineCard() {
  return (
    <PanelFrame title="Tests" state="ready">
      <div className="pipeline">
        <GithubActions />
        <LocalRun />
      </div>
    </PanelFrame>
  );
}
