import { useEffect, useLayoutEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { PanelFrame } from "../components/PanelFrame";
import { StatusWord, type StatusKind } from "../components/StatusWord";
import { panelState } from "../components/bits";
import { apiSend, errorMessage } from "../lib/api";
import { useJournal } from "../lib/useAgentLog";
import { AGENTS_REFRESH_MS, useResource } from "../lib/useResource";
import type { AgentEvent, AgentState, ChiefSummary } from "../../../shared/types";

const MESSAGE_MAX = 8000;
const TAIL_SLACK_PX = 6;
const TOOL_PREFIX = "mcp__dashboard__";
const PREVIEW_CHARS = 80;

/** One line of a tool result for its collapsed summary; the full text is one click away. */
function preview(output: string): string {
  const first = output.trim().split("\n", 1)[0] || "(empty)";
  return first.length > PREVIEW_CHARS ? `${first.slice(0, PREVIEW_CHARS)}…` : first;
}

const WORDS: Partial<Record<AgentState, { kind: StatusKind; text: string }>> = {
  running: { kind: "running", text: "Working" },
  "waiting-approval": { kind: "attention", text: "Waiting for approval" },
  idle: { kind: "idle", text: "Idle" },
  failed: { kind: "fail", text: "Failed" },
  stopped: { kind: "idle", text: "Stopped" },
};


function Line({ e }: { e: AgentEvent }) {
  switch (e.kind) {
    case "user":
      return <p className="chat__row chat__row--user"><span className="alog__who">You</span> <span className="alog__text">{e.text}</span></p>;
    case "text":
      return <p className="chat__row"><span className="alog__who">Chief</span> <span className="alog__text">{e.text}</span></p>;
    case "tool-use":
      return (
        <div className="chat__tool">
          <span className="mono chat__chip">{e.tool.startsWith(TOOL_PREFIX) ? e.tool.slice(TOOL_PREFIX.length) : e.tool}</span>
          {Object.keys(e.input).length > 0 && <pre className="mono alog__out">{JSON.stringify(e.input, null, 2)}</pre>}
        </div>
      );
    case "tool-result":
      // Results are often thousands of characters of raw tool output: collapsed they stay out of the way of the replies; failures start open.
      return (
        <details className={`chat__result dim${e.isError ? " alog__err" : ""}`} open={e.isError}>
          <summary>
            {e.isError ? "Failed" : "Result"} <span className="mono">{preview(e.output)}</span>
          </summary>
          <pre className={`mono alog__out${e.isError ? " alog__out--err" : ""}`}>{e.output}</pre>
        </details>
      );
    case "approval":
      return e.outcome === "requested" ? null : <p className="chat__row dim">{e.outcome === "allow" ? "You approved it." : e.outcome === "deny" ? "You denied it." : "It was cancelled."}</p>;
    case "error":
      return <p className="chat__row alog__err">{e.message}</p>;
    default:
      return null; // state changes and turn results are shown by the status line instead
  }
}

export function ChiefChat() {
  const res = useResource<ChiefSummary>("chat", "/api/chief", () => false, AGENTS_REFRESH_MS);
  const log = useJournal("/api/chief/events", "chat", true);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [confirmReset, setConfirmReset] = useState(false);
  const [announce, setAnnounce] = useState("");
  const box = useRef<HTMLDivElement>(null);
  const atTail = useRef(true);
  const spoken = useRef(0);
  const s = res.data;

  // The box only exists once the panel has left its loading state, so mounting it counts as a reason to follow the tail too.
  const ready = panelState(res) === "ready";
  useLayoutEffect(() => {
    const el = box.current;
    if (el && atTail.current) el.scrollTop = el.scrollHeight;
  }, [log.events.length, ready]);

  // Announce a reply once it is complete (the chief is idle again), never while it is still working.
  useEffect(() => {
    if (!s || s.state === "running") return;
    const lastText = [...log.events].reverse().find((e) => e.kind === "text");
    if (lastText && lastText.seq > spoken.current && lastText.kind === "text") {
      spoken.current = lastText.seq;
      setAnnounce(`Chief of staff: ${lastText.text}`);
    }
  }, [log.events, s]);

  async function post(path: string, body?: unknown): Promise<boolean> {
    setBusy(true);
    setError(null);
    try {
      await apiSend("POST", path, body);
      res.reload();
      return true;
    } catch (err) {
      setError(errorMessage(err, "The request failed."));
      return false;
    } finally {
      setBusy(false);
    }
  }

  async function send(ev?: FormEvent) {
    ev?.preventDefault();
    if (!text.trim() || busy) return;
    if (await post("/api/chief/message", { text })) setText("");
  }
  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      void send();
    }
  };

  const word = s ? WORDS[s.state] : undefined;
  const working = s?.state === "running" || s?.state === "waiting-approval";
  return (
    <PanelFrame title="Chief of staff" state={panelState(res)} error={res.error ?? undefined} onRetry={res.reload}>
      {s && (
        <div className="row row--wrap">
          {word && <StatusWord kind={word.kind}>{word.text}</StatusWord>}
          {s.pendingApprovals > 0 && <a href="#approvals" className="agent__pending">{s.pendingApprovals} pending</a>}
          <span className="dim mono">{s.model}</span>
          {working && <button type="button" className="btn" disabled={busy} onClick={() => void post("/api/chief/stop")}>Stop</button>}
          {!confirmReset ? (
            <button type="button" className="btn btn--quiet" disabled={busy || (s.turns === 0 && !s.sessionId && s.state !== "failed")} onClick={() => setConfirmReset(true)}>New conversation</button>
          ) : (
            <>
              <button type="button" className="btn" disabled={busy} onClick={() => void post("/api/chief/reset").then(() => setConfirmReset(false))}>Yes, start over</button>
              <button type="button" className="btn btn--quiet" onClick={() => setConfirmReset(false)}>Keep it</button>
            </>
          )}
        </div>
      )}
      {s?.error && <p className="approval__error">{s.error}</p>}
      <p className="sr-only" role="status" aria-live="polite">{announce}</p>
      <div
        ref={box}
        className="alog chat"
        role="log"
        aria-label="Conversation with the chief of staff"
        tabIndex={0}
        onScroll={(e) => {
          const el = e.currentTarget;
          atTail.current = el.scrollHeight - el.scrollTop - el.clientHeight <= TAIL_SLACK_PX;
        }}
      >
        {log.loaded && log.events.length === 0 && <p className="dim">Ask the chief of staff to look at the repository or to start workers. Anything that changes something waits for your approval.</p>}
        {log.events.map((e) => <Line key={e.seq} e={e} />)}
      </div>
      {log.error && <p className="approval__error" role="alert">{log.error}</p>}
      {error && <p className="approval__error" role="alert">{error}</p>}
      <form className="agent__send" onSubmit={(ev) => void send(ev)}>
        <label className="sr-only" htmlFor="chief-msg">Message to the chief of staff</label>
        <textarea id="chief-msg" className="field" rows={2} maxLength={MESSAGE_MAX} value={text} onChange={(e) => setText(e.target.value)} onKeyDown={onKey} placeholder="Message the chief of staff (Ctrl+Enter sends)" />
        <button type="submit" className="btn btn--primary" disabled={busy || !text.trim()}>Send</button>
      </form>
    </PanelFrame>
  );
}
