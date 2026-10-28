import { useEffect, useLayoutEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { ConversationList } from "../components/ConversationList";
import { PanelFrame } from "../components/PanelFrame";
import { StatusWord, type StatusKind } from "../components/StatusWord";
import { panelState } from "../components/bits";
import { apiSend, errorMessage } from "../lib/api";
import { useJournal } from "../lib/useAgentLog";
import { AGENTS_REFRESH_MS, useResource } from "../lib/useResource";
import type { AgentEvent, AgentState, ChiefSummary, ConversationInfo, ConversationsResponse } from "../../../shared/types";

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

interface PaneProps {
  open: boolean;
  viewing: string | null;
  busy: boolean;
  working: boolean;
  onRows: (rows: ConversationInfo[] | null) => void;
  onView: (id: string) => void;
  onResume: (id: string) => void;
  onRename: (id: string, title: string) => void;
  onArchive: (id: string, archived: boolean) => void;
}

/** Loads the conversations only while they are needed: the list is open, or one is being read (to know whether it can be resumed). */
function ConversationsPane({ open, onRows, ...list }: PaneProps) {
  const convs = useResource<ConversationsResponse>("chat", "/api/chief/conversations", () => false, AGENTS_REFRESH_MS);
  const rows = convs.data?.conversations ?? null;
  useEffect(() => {
    onRows(rows);
  }, [rows, onRows]);
  return open ? <ConversationList rows={rows ?? []} viewingId={list.viewing} busy={list.busy} locked={list.working} onView={list.onView} onResume={list.onResume} onRename={list.onRename} onArchive={list.onArchive} /> : null;
}

export function ChiefChat() {
  const res = useResource<ChiefSummary>("chat", "/api/chief", () => false, AGENTS_REFRESH_MS);
  const s = res.data;
  const [rows, setRows] = useState<ConversationInfo[] | null>(null);
  const [listOpen, setListOpen] = useState(false);
  const [viewId, setViewId] = useState<string | null>(null);
  // A parked conversation open for reading; the live one is shown whenever this is null.
  const viewing = viewId !== null && s !== null && viewId !== s.conversationId ? viewId : null;
  // The conversation being read may have become the active one some other way; do not keep it as a pending view.
  useEffect(() => {
    if (viewId !== null && s !== null && viewId === s.conversationId) setViewId(null);
  }, [viewId, s]);
  const log = useJournal(viewing ? `/api/chief/conversations/${encodeURIComponent(viewing)}/events` : "/api/chief/events", "chat", true);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [confirmReset, setConfirmReset] = useState(false);
  const [announce, setAnnounce] = useState("");
  const box = useRef<HTMLDivElement>(null);
  const atTail = useRef(true);
  const spoken = useRef(0);
  const spokenIn = useRef<string | null>(null);
  const seenView = useRef<string | null>(null);

  // The box only exists once the panel has left its loading state, so mounting it counts as a reason to follow the tail too.
  const ready = panelState(res) === "ready";
  useLayoutEffect(() => {
    const el = box.current;
    if (seenView.current !== viewing) {
      seenView.current = viewing;
      atTail.current = true; // another log opens at its newest entry, wherever the last one was scrolled
    }
    if (el && atTail.current) el.scrollTop = el.scrollHeight;
  }, [log.events.length, ready, viewing]);

  // Announce a reply once it is complete (the chief is idle again), never while it is still working.
  useEffect(() => {
    if (!s || s.state === "running" || viewing) return; // an earlier conversation being read is not a new reply
    const lastText = [...log.events].reverse().find((e) => e.kind === "text");
    if (spokenIn.current !== null && spokenIn.current !== s.conversationId) {
      // Another conversation became the active one: its existing replies are not new, and its numbering starts over.
      spokenIn.current = s.conversationId;
      spoken.current = lastText?.seq ?? 0;
      return;
    }
    spokenIn.current = s.conversationId;
    if (lastText && lastText.seq > spoken.current && lastText.kind === "text") {
      spoken.current = lastText.seq;
      setAnnounce(`Chief of staff: ${lastText.text}`);
    }
  }, [log.events, s, viewing]);

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

  async function patch(id: string, body: { title?: string; archived?: boolean }) {
    setBusy(true);
    setError(null);
    try {
      await apiSend("PATCH", `/api/chief/conversations/${encodeURIComponent(id)}`, body);
      res.reload(); // the list refreshes itself: the server announces the change
    } catch (err) {
      setError(errorMessage(err, "The request failed."));
    } finally {
      setBusy(false);
    }
  }
  async function resume(id: string) {
    if (await post(`/api/chief/conversations/${encodeURIComponent(id)}/resume`)) {
      setViewId(null);
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
  const shown = viewing ? rows?.find((c) => c.id === viewing) : undefined;
  const frozen = viewing ? shown?.readOnly ?? null : s?.readOnly ?? null;
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
          <button type="button" className="btn btn--quiet" aria-expanded={listOpen} onClick={() => setListOpen((o) => !o)}>Conversations</button>
        </div>
      )}
      {(listOpen || viewing) && (
        <ConversationsPane
          open={listOpen}
          viewing={viewing}
          busy={busy}
          working={working}
          onRows={setRows}
          onView={(id) => setViewId(id === s?.conversationId ? null : id)}
          onResume={(id) => void resume(id)}
          onRename={(id, title) => void patch(id, { title })}
          onArchive={(id, archived) => void patch(id, { archived })}
        />
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
      {viewing || frozen ? (
        <div className="chat__bar" role="status">
          <span>{frozen ? "This conversation can be read but not continued." : "You are reading an earlier conversation."}</span>
          {viewing && shown && !frozen && <button type="button" className="btn btn--primary" disabled={busy || working} title={working ? "Stop the chief of staff first: switching ends its current turn." : undefined} onClick={() => void resume(viewing)}>Resume this conversation</button>}
          {viewing && <button type="button" className="btn btn--quiet" onClick={() => setViewId(null)}>Back to current</button>}
        </div>
      ) : (
        <form className="agent__send" onSubmit={(ev) => void send(ev)}>
          <label className="sr-only" htmlFor="chief-msg">Message to the chief of staff</label>
          <textarea id="chief-msg" className="field" rows={2} maxLength={MESSAGE_MAX} value={text} onChange={(e) => setText(e.target.value)} onKeyDown={onKey} placeholder="Message the chief of staff (Ctrl+Enter sends)" />
          <button type="submit" className="btn btn--primary" disabled={busy || !text.trim()}>Send</button>
        </form>
      )}
    </PanelFrame>
  );
}
