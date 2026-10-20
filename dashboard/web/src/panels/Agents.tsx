import { useLayoutEffect, useRef, useState, type FormEvent } from "react";
import { CaretDown, CaretRight, Plus } from "@phosphor-icons/react";
import { PanelFrame } from "../components/PanelFrame";
import { StatusWord, type StatusKind } from "../components/StatusWord";
import { panelState } from "../components/bits";
import { ApiError, apiSend } from "../lib/api";
import { relTime } from "../lib/time";
import { useAgentLog } from "../lib/useAgentLog";
import { AGENTS_REFRESH_MS, useResource } from "../lib/useResource";
import type { AgentEvent, AgentState, AgentSummary, AgentsResponse } from "../../../shared/types";

const NAME_RE = /^[a-z0-9][a-z0-9-]{1,30}$/;
const NAME_HELP = "Name must be 2 to 31 characters: lowercase letters, digits and hyphens.";
const TASK_MAX = 8000;
const MESSAGE_MAX = 8000;
const TAIL_SLACK_PX = 6;
const ACTIVE: AgentState[] = ["starting", "running", "waiting-approval", "idle"];
const FINAL: AgentState[] = ["done", "failed", "stopped"];

const WORDS: Record<AgentState, { kind: StatusKind; text: string }> = {
  starting: { kind: "running", text: "Starting" },
  running: { kind: "running", text: "Running" },
  "waiting-approval": { kind: "attention", text: "Waiting for approval" },
  idle: { kind: "idle", text: "Idle" },
  done: { kind: "ok", text: "Done" },
  failed: { kind: "fail", text: "Failed" },
  stopped: { kind: "idle", text: "Stopped" },
};

const message = (err: unknown, fallback: string) => (err instanceof ApiError ? err.message : fallback);

function Entry({ e }: { e: AgentEvent }) {
  switch (e.kind) {
    case "user":
      return <p className="alog__row"><span className="alog__who">You</span> <span className="alog__text">{e.text}</span></p>;
    case "text":
      return <p className="alog__row"><span className="alog__who">Agent</span> <span className="alog__text">{e.text}</span></p>;
    case "tool-use":
      return (
        <p className="alog__row">
          <span className="alog__who dim">{e.tool}</span>{" "}
          <code className="mono alog__text">{typeof e.input.command === "string" ? e.input.command : typeof e.input.file_path === "string" ? e.input.file_path : JSON.stringify(e.input)}</code>
        </p>
      );
    case "tool-result":
      return <pre className={`mono alog__out${e.isError ? " alog__out--err" : ""}`}>{e.output}</pre>;
    case "approval":
      return <p className="alog__row dim">{e.outcome === "requested" ? `Approval requested for ${e.tool}` : `Approval ${e.outcome === "allow" ? "allowed" : e.outcome === "deny" ? "denied" : "cancelled"} for ${e.tool}`}</p>;
    case "state":
      return <p className="alog__row dim">State: {e.state}{e.detail ? ` (${e.detail})` : ""}</p>;
    case "result":
      return <p className="alog__row dim">{e.ok ? "Finished a turn" : "Turn ended early"}{e.costUsd !== null ? ` · about $${e.costUsd.toFixed(2)}` : ""}</p>;
    case "error":
      return <p className="alog__row alog__err">{e.message}</p>;
  }
}

function AgentDetail({ a, onChanged }: { a: AgentSummary; onChanged: () => void }) {
  const log = useAgentLog(a.id, true);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const box = useRef<HTMLDivElement>(null);
  const atTail = useRef(true);
  const active = ACTIVE.includes(a.state);

  useLayoutEffect(() => {
    const el = box.current;
    if (el && atTail.current) el.scrollTop = el.scrollHeight;
  }, [log.events.length]);

  async function act(path: string, body?: unknown): Promise<boolean> {
    setBusy(true);
    setError(null);
    try {
      await apiSend("POST", `/api/agents/${encodeURIComponent(a.id)}/${path}`, body);
      onChanged();
      return true;
    } catch (err) {
      setError(message(err, "The request failed."));
      return false;
    } finally {
      setBusy(false);
    }
  }

  async function send(ev: FormEvent) {
    ev.preventDefault();
    if (!text.trim() || busy) return;
    if (await act("message", { text })) setText("");
  }

  return (
    <div className="agent__detail">
      <div
        ref={box}
        className="alog"
        role="log"
        aria-label={`Log of ${a.name}`}
        tabIndex={0}
        onScroll={(e) => {
          const el = e.currentTarget;
          atTail.current = el.scrollHeight - el.scrollTop - el.clientHeight <= TAIL_SLACK_PX;
        }}
      >
        {!log.loaded && <p className="dim">Loading log</p>}
        {log.loaded && log.events.length === 0 && !log.error && <p className="dim">Nothing logged yet.</p>}
        {log.events.map((e) => <Entry key={e.seq} e={e} />)}
      </div>
      {log.error && <p className="approval__error" role="alert">{log.error}</p>}
      {error && <p className="approval__error" role="alert">{error}</p>}
      <div className="row row--wrap">
        {active && <button type="button" className="btn" disabled={busy} onClick={() => void act("stop")}>Stop</button>}
        {FINAL.includes(a.state) && a.sessionId && <button type="button" className="btn" disabled={busy} onClick={() => void act("resume")}>Resume</button>}
        <span className="dim mono">{a.branch}</span>
      </div>
      {active && (
        <form className="agent__send" onSubmit={(ev) => void send(ev)}>
          <label className="sr-only" htmlFor={`msg-${a.id}`}>Message to {a.name}</label>
          <textarea id={`msg-${a.id}`} className="field" rows={2} maxLength={MESSAGE_MAX} value={text} onChange={(e) => setText(e.target.value)} placeholder={`Message to ${a.name}`} />
          <button type="submit" className="btn btn--primary" disabled={busy || !text.trim()}>Send</button>
        </form>
      )}
    </div>
  );
}

function NewAgent({ onStarted, onCancel }: { onStarted: () => void; onCancel: () => void }) {
  const [name, setName] = useState("");
  const [task, setTask] = useState("");
  const [issue, setIssue] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(ev: FormEvent) {
    ev.preventDefault();
    if (busy) return;
    const n = name.trim();
    if (!NAME_RE.test(n)) return setError(NAME_HELP);
    if (!task.trim()) return setError("Task is required.");
    if (issue.trim() !== "" && !/^[1-9]\d{0,9}$/.test(issue.trim())) return setError("Issue must be a whole number.");
    setBusy(true);
    setError(null);
    try {
      await apiSend("POST", "/api/agents", { name: n, task, ...(issue.trim() ? { issue: Number(issue.trim()) } : {}) });
      onStarted();
    } catch (err) {
      setError(message(err, "Could not start the agent."));
    } finally {
      setBusy(false);
    }
  }

  return (
    <form className="agent__new" onSubmit={(ev) => void submit(ev)} noValidate>
      <label className="field__label" htmlFor="agent-name">Name</label>
      <input id="agent-name" className="field mono" autoFocus value={name} onChange={(e) => setName(e.target.value)} autoComplete="off" spellCheck={false} />
      <label className="field__label" htmlFor="agent-task">Task</label>
      <textarea id="agent-task" className="field" rows={4} maxLength={TASK_MAX} value={task} onChange={(e) => setTask(e.target.value)} />
      <label className="field__label" htmlFor="agent-issue">Issue number (optional)</label>
      <input id="agent-issue" className="field mono" inputMode="numeric" value={issue} onChange={(e) => setIssue(e.target.value)} autoComplete="off" />
      {error && <p className="approval__error" role="alert">{error}</p>}
      <div className="row">
        <button type="submit" className="btn btn--primary" disabled={busy}>Start agent</button>
        <button type="button" className="btn btn--quiet" onClick={onCancel}>Cancel</button>
      </div>
    </form>
  );
}

export function Agents() {
  const res = useResource<AgentsResponse>("agents", "/api/agents", (d) => d.agents.length === 0, AGENTS_REFRESH_MS);
  const [adding, setAdding] = useState(false);
  const [open, setOpen] = useState<string | null>(null);
  const addButton = useRef<HTMLButtonElement>(null);
  const agents = res.data?.agents ?? [];
  const running = agents.filter((a) => ACTIVE.includes(a.state)).length;

  const closeForm = () => {
    setAdding(false);
    addButton.current?.focus();
  };

  return (
    <PanelFrame
      title="Agents"
      count={agents.length}
      state={panelState(res)}
      error={res.error ?? undefined}
      onRetry={res.reload}
      actions={
        <button ref={addButton} type="button" className="btn" aria-expanded={adding} onClick={() => setAdding((v) => !v)}>
          <Plus size={14} aria-hidden="true" /> New agent
        </button>
      }
    >
      {res.data && <p className="dim mono">{running} / {res.data.max} running</p>}
      {adding && <NewAgent onCancel={closeForm} onStarted={() => { closeForm(); res.reload(); }} />}
      {agents.length === 0 ? (
        <p className="dim">No agents yet. Start one with New agent; each works in its own git worktree.</p>
      ) : (
        <ul className="agents">
          {agents.map((a) => {
            const w = WORDS[a.state];
            const isOpen = open === a.id;
            return (
              <li key={a.id} className="agent">
                <div className="agent__head">
                  <button type="button" className="btn btn--bare" aria-expanded={isOpen} onClick={() => setOpen(isOpen ? null : a.id)}>
                    {isOpen ? <CaretDown size={14} aria-hidden="true" /> : <CaretRight size={14} aria-hidden="true" />}
                    <span className="mono">{a.name}</span>
                  </button>
                  <StatusWord kind={w.kind}>{w.text}</StatusWord>
                  {a.pendingApprovals > 0 && <a href="#approvals" className="agent__pending">{a.pendingApprovals} pending</a>}
                  <span className="dim agent__when">{relTime(a.updatedAt)}</span>
                </div>
                <p className="dim agent__task">{a.task}</p>
                {a.error && <p className="agent__error">{a.error}</p>}
                {isOpen && <AgentDetail a={a} onChanged={res.reload} />}
              </li>
            );
          })}
        </ul>
      )}
    </PanelFrame>
  );
}
