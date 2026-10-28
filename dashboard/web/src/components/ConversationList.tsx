import { useState, type KeyboardEvent } from "react";
import type { ConversationInfo } from "../../../shared/types";

const TITLE_MAX = 120;
const usd = (n: number) => `about $${n.toFixed(2)}`;
const when = (iso: string) => new Date(iso).toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
const label = (r: ConversationInfo) => r.title ?? "Untitled";

export interface ConversationListProps {
  rows: ConversationInfo[];
  /** The parked conversation whose log is open in the panel, if any. */
  viewingId: string | null;
  busy: boolean;
  /** Switching conversations ends the current turn, so Resume waits until the chief is not working. */
  locked?: boolean;
  onView: (id: string) => void;
  onResume: (id: string) => void;
  onRename: (id: string, title: string) => void;
  onArchive: (id: string, archived: boolean) => void;
}

/** The chief's conversations: read, resume, rename, archive. Presentational; the panel owns the requests. */
export function ConversationList(p: ConversationListProps) {
  const [showArchived, setShowArchived] = useState(false);
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const rows = p.rows.filter((r) => r.archived === showArchived);

  const commit = (id: string) => {
    const title = draft.trim();
    setEditing(null);
    if (title && title !== (p.rows.find((r) => r.id === id)?.title ?? null)) p.onRename(id, title);
  };
  const onKey = (e: KeyboardEvent<HTMLInputElement>, id: string) => {
    if (e.key === "Enter") {
      e.preventDefault();
      commit(id);
    } else if (e.key === "Escape") {
      e.stopPropagation(); // cancelling an edit must not also close the drawer around the list
      setEditing(null);
    }
  };

  return (
    <section className="convs" aria-label="Conversations">
      <label className="convs__toggle">
        <input type="checkbox" checked={showArchived} onChange={(e) => setShowArchived(e.target.checked)} /> Show archived
      </label>
      {rows.length === 0 && <p className="dim">{showArchived ? "No archived conversations." : "No conversations yet."}</p>}
      <ul className="convs__list">
        {rows.map((r) => (
          <li key={r.id} className={`convs__row${p.viewingId === r.id ? " convs__row--open" : ""}`}>
            {editing === r.id ? (
              <input
                className="field"
                aria-label="Conversation title"
                autoFocus
                maxLength={TITLE_MAX}
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={(e) => onKey(e, r.id)}
                onBlur={() => setEditing(null)}
              />
            ) : (
              <span className="convs__title">{label(r)}</span>
            )}
            <span className="dim mono convs__meta">
              {when(r.updatedAt)} · {r.turns} {r.turns === 1 ? "turn" : "turns"}
              {r.costUsd !== null ? ` · ${usd(r.costUsd)}` : ""}
            </span>
            {r.active && <span className="convs__badge">Current</span>}
            {r.readOnly && <span className="convs__badge">Read-only</span>}
            <span className="convs__actions">
              {!r.active && <button type="button" className="btn btn--quiet" aria-label={`Read ${label(r)}`} disabled={p.busy} onClick={() => p.onView(r.id)}>Read</button>}
              {!r.active && !r.readOnly && <button type="button" className="btn btn--quiet" aria-label={`Resume ${label(r)}`} disabled={p.busy || p.locked} title={p.locked ? "Stop the chief of staff first: switching ends its current turn." : undefined} onClick={() => p.onResume(r.id)}>Resume</button>}
              <button type="button" className="btn btn--quiet" aria-label={`Rename ${label(r)}`} disabled={p.busy} onMouseDown={(e) => e.preventDefault()} onClick={() => { setEditing(r.id); setDraft(r.title ?? ""); }}>Rename</button>
              {!r.active && <button type="button" className="btn btn--quiet" aria-label={`${r.archived ? "Restore" : "Archive"} ${label(r)}`} disabled={p.busy} onClick={() => p.onArchive(r.id, !r.archived)}>{r.archived ? "Restore" : "Archive"}</button>}
            </span>
          </li>
        ))}
      </ul>
    </section>
  );
}
