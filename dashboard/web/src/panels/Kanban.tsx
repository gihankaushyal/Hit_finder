import { useCallback, useEffect, useId, useMemo, useRef, useState, type FormEvent } from "react";
import { ArrowsClockwise, Plus } from "@phosphor-icons/react";
import { PanelFrame } from "../components/PanelFrame";
import { StatusWord } from "../components/StatusWord";
import { ExternalLink, StaleNotice, panelState } from "../components/bits";
import { ApiError, apiSend } from "../lib/api";
import { relTime } from "../lib/time";
import { useResource } from "../lib/useResource";
import type { Board, Task, TaskStatus } from "../../../shared/types";

/** The Done column shows this many of the most recently updated tasks until asked for all. */
export const DONE_VISIBLE_COUNT = 30;
// Mirrors the server limit (server/routes/kanban.ts).
export const TASK_TITLE_MAX_CHARS = 200;
const IMPORT_PREVIEW_COMMAND = "npm run kanban:sync -- --dry-run";

const COLUMNS: { status: TaskStatus; label: string }[] = [
  { status: "todo", label: "Todo" },
  { status: "in-progress", label: "In progress" },
  { status: "blocked", label: "Blocked" },
  { status: "done", label: "Done" },
];
const LABELS = Object.fromEntries(COLUMNS.map((c) => [c.status, c.label])) as Record<TaskStatus, string>;

interface Override {
  status: TaskStatus;
  seq: number;
  /** The server accepted it; drop the override once fresh board data has arrived. */
  settled: boolean;
}
type Control = "check" | "move";

const isEmpty = (b: Board) => COLUMNS.every((c) => b.columns[c.status].length === 0);

function AddTask({ onAdded }: { onAdded: () => void }) {
  const id = useId();
  const errId = useId();
  const ref = useRef<HTMLInputElement>(null);
  const [title, setTitle] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (ev: FormEvent) => {
    ev.preventDefault();
    if (pending) return;
    const t = title.trim();
    if (t === "") return setError("Enter a task title.");
    if (t.length > TASK_TITLE_MAX_CHARS) return setError(`Task must be at most ${TASK_TITLE_MAX_CHARS} characters.`);
    setError(null);
    setPending(true);
    try {
      await apiSend("POST", "/api/kanban/tasks", { title: t });
      setTitle("");
      onAdded();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not add the task.");
    } finally {
      setPending(false);
      ref.current?.focus();
    }
  };

  return (
    <form className="kb-add" onSubmit={submit} noValidate>
      <label htmlFor={id}>Task</label>
      <div className="kb-add__row">
        <input id={id} ref={ref} name="task-title" autoComplete="off" value={title} maxLength={TASK_TITLE_MAX_CHARS} onChange={(e) => setTitle(e.target.value)}
          aria-invalid={error !== null} aria-describedby={error ? errId : undefined} />
        <button type="submit" className="btn" disabled={pending}>
          <Plus size={14} aria-hidden="true" />
          <span>Add task</span>
        </button>
      </div>
      {error && <p id={errId} className="field__error">{error}</p>}
    </form>
  );
}

function TaskRow({ task, editable, error, conflicts, onChange }: { task: Task; editable: boolean; error?: string; conflicts: string[]; onChange: (t: Task, s: TaskStatus, c: Control) => void }) {
  const done = task.status === "done";
  const title = done ? <s className="dim">{task.title}</s> : <span>{task.title}</span>;
  return (
    <li className="kb-item">
      <div className="kb-item__main">
        {editable && (
          <input
            type="checkbox" id={`kb-${task.number}-check`} checked={done}
            aria-label={`Mark ${task.title} done`}
            onChange={() => onChange(task, done ? "todo" : "done", "check")}
          />
        )}
        <span className="kb-item__title">{title}</span>
      </div>
      <div className="kb-item__meta">
        {task.number > 0 && (editable && task.url ? (
          <ExternalLink href={task.url} className="mono">
            <span aria-hidden="true">#{task.number}</span>
            <span className="sr-only">Issue #{task.number}, {task.title}</span>
          </ExternalLink>
        ) : (
          <span className="mono dim">#{task.number}</span>
        ))}
        {task.kind && <span className="chip">{task.kind}</span>}
        {!task.inFile && <span className="chip">not in file</span>}
        {editable && (
          <select
            id={`kb-${task.number}-move`} className="kb-item__move" aria-label={`Move ${task.title}`} value={task.status}
            onChange={(e) => onChange(task, e.target.value as TaskStatus, "move")}
          >
            {COLUMNS.map((c) => <option key={c.status} value={c.status}>{c.label}</option>)}
          </select>
        )}
      </div>
      {conflicts.map((c, i) => (
        <p key={i} className="kb-item__conflict">
          <StatusWord kind="attention">Conflict</StatusWord> <span>{c}</span>
        </p>
      ))}
      {error && <p className="field__error" role="alert">{error}</p>}
    </li>
  );
}

export function Kanban() {
  const res = useResource<Board>("kanban", "/api/kanban", isEmpty);
  const board = res.data;
  const editable = board?.imported === true;
  const [overrides, setOverrides] = useState<Record<number, Override>>({});
  const [errors, setErrors] = useState<Record<number, string>>({});
  const [showAllDone, setShowAllDone] = useState(false);
  const [syncing, setSyncing] = useState(false);
  const [syncError, setSyncError] = useState<string | null>(null);

  const seq = useRef(0);
  const latestSeq = useRef(new Map<number, number>());
  // Requests for one task run one after another, so the server applies them in the order chosen.
  const chains = useRef(new Map<number, Promise<void>>());
  const focusAfter = useRef<{ number: number; control: Control } | null>(null);

  const { reload } = res;
  useEffect(() => {
    // Fresh data has arrived: overrides the server already accepted are no longer needed.
    setOverrides((o) => {
      const keep = Object.fromEntries(Object.entries(o).filter(([, v]) => !v.settled));
      return Object.keys(keep).length === Object.keys(o).length ? o : keep;
    });
  }, [board]);

  useEffect(() => {
    const f = focusAfter.current;
    if (!f) return;
    const el = document.getElementById(`kb-${f.number}-${f.control}`);
    if (el) {
      focusAfter.current = null;
      if (document.activeElement !== el) el.focus();
    }
  });

  const change = useCallback(
    (task: Task, status: TaskStatus, control: Control) => {
      if (status === task.status) return;
      const mine = ++seq.current;
      latestSeq.current.set(task.number, mine);
      focusAfter.current = { number: task.number, control };
      setErrors((e) => {
        const { [task.number]: _drop, ...rest } = e;
        return rest;
      });
      setOverrides((o) => ({ ...o, [task.number]: { status, seq: mine, settled: false } }));
      const isLatest = () => latestSeq.current.get(task.number) === mine;
      const previous = chains.current.get(task.number) ?? Promise.resolve();
      const next = previous.then(async () => {
        try {
          await apiSend("PATCH", `/api/kanban/tasks/${task.number}`, { status });
          if (isLatest()) setOverrides((o) => (o[task.number]?.seq === mine ? { ...o, [task.number]: { ...o[task.number], settled: true } } : o));
        } catch (err) {
          if (isLatest()) {
            setOverrides((o) => {
              const { [task.number]: _drop, ...rest } = o;
              return rest;
            });
            const why = err instanceof ApiError ? err.message : "Request failed";
            setErrors((e) => ({ ...e, [task.number]: `Could not move "${task.title}" to ${LABELS[status]}: ${why}. It was moved back.` }));
          }
        }
        if (isLatest()) reload();
      });
      chains.current.set(task.number, next);
    },
    [reload],
  );

  const columns = useMemo(() => {
    const out: Record<TaskStatus, Task[]> = { todo: [], "in-progress": [], blocked: [], done: [] };
    if (!board) return out;
    for (const c of COLUMNS) {
      for (const t of board.columns[c.status]) {
        const o = overrides[t.number];
        const effective = o ? o.status : t.status;
        out[effective].push(effective === t.status ? t : { ...t, status: effective });
      }
    }
    // A task just completed goes to the top of Done; the server sorts the rest by recency.
    out.done.sort((a, b) => Number(overrides[b.number]?.status === "done") - Number(overrides[a.number]?.status === "done"));
    for (const s of ["todo", "in-progress", "blocked"] as const) out[s].sort((a, b) => a.number - b.number);
    return out;
  }, [board, overrides]);

  const sync = async () => {
    setSyncing(true);
    setSyncError(null);
    try {
      await apiSend("POST", "/api/kanban/sync");
    } catch (err) {
      setSyncError(err instanceof ApiError ? err.message : "Sync failed.");
    } finally {
      setSyncing(false);
      reload();
    }
  };

  const conflictsByIssue = useMemo(() => {
    const m = new Map<number, string[]>();
    for (const c of board?.conflictItems ?? []) if (c.issue !== null) m.set(c.issue, [...(m.get(c.issue) ?? []), c.message]);
    return m;
  }, [board]);

  const allTasks = board ? COLUMNS.flatMap((c) => board.columns[c.status]) : [];
  const notInFile = allTasks.filter((t) => !t.inFile).length;
  const actions =
    board && editable ? (
      <>
        {board.lastSyncAt && (
          <span className="dim kb-sync">Synced <time dateTime={board.lastSyncAt}>{relTime(board.lastSyncAt)}</time></span>
        )}
        <button type="button" className="btn" disabled={syncing} onClick={sync}>
          <ArrowsClockwise size={14} aria-hidden="true" className={syncing ? "spin" : undefined} />
          <span>{syncing ? "Syncing…" : "Sync now"}</span>
        </button>
      </>
    ) : undefined;

  return (
    <PanelFrame title="Kanban" actions={actions} state={panelState(res)} error={res.error ?? undefined} onRetry={res.reload}>
      <StaleNotice res={res} label="the board" />
      {board && !editable && (
        <div className="notice" role="note">
          <span>
            Tasks are read from the kanban file. Syncing with GitHub starts after the first import; preview it with{" "}
            <code className="mono">{IMPORT_PREVIEW_COMMAND}</code>
          </span>
        </div>
      )}
      {syncError && <p className="field__error" role="alert">{syncError}</p>}
      {board && board.syncError && (
        <div className="notice notice--fail">
          <StatusWord kind="fail">Sync problem</StatusWord>
          <span>{board.syncError}</span>
        </div>
      )}
      {board && board.conflicts.length > 0 && (
        <details className="notice notice--act">
          <summary>
            <StatusWord kind="attention">{board.conflicts.length} {board.conflicts.length === 1 ? "conflict needs" : "conflicts need"} a look</StatusWord>
          </summary>
          <ul className="bullets">
            {board.conflicts.map((c, i) => <li key={i}>{c}</li>)}
          </ul>
        </details>
      )}
      {notInFile > 0 && <p className="dim kb-note">{notInFile} {notInFile === 1 ? "task is" : "tasks are"} on GitHub but not in the kanban file.</p>}
      {board && isEmpty(board) && (
        <p className="dim kb-note">
          {editable ? "No tasks yet. Add one in the Todo column." : "No tasks found in the kanban file yet. Add some there, then run the first import."}
        </p>
      )}
      {board && (
        <div className="kb">
          {COLUMNS.map(({ status, label }) => {
            const tasks = columns[status];
            const capped = status === "done" && !showAllDone && tasks.length > DONE_VISIBLE_COUNT;
            const shown = capped ? tasks.slice(0, DONE_VISIBLE_COUNT) : tasks;
            return (
              <section key={status} className="kb-col" aria-labelledby={`kb-h-${status}`}>
                <h3 id={`kb-h-${status}`} className="kb-col__head">
                  {label} <span className="mono dim">{tasks.length}</span>
                </h3>
                {status === "todo" && editable && <AddTask onAdded={reload} />}
                <ul className="plain kb-col__list">
                  {shown.map((t, i) => (
                    <TaskRow key={`${t.number}-${i}`} task={t} editable={editable} error={errors[t.number]} conflicts={conflictsByIssue.get(t.number) ?? []} onChange={change} />
                  ))}
                </ul>
                {tasks.length === 0 && <p className="dim">Nothing here.</p>}
                {status === "done" && tasks.length > DONE_VISIBLE_COUNT && (
                  <button type="button" className="btn btn--quiet" onClick={() => setShowAllDone(!showAllDone)}>
                    {showAllDone ? "Show fewer" : `Show all (${tasks.length})`}
                  </button>
                )}
              </section>
            );
          })}
        </div>
      )}
    </PanelFrame>
  );
}
