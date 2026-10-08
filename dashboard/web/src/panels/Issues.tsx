import { useId, useRef, useState, type FormEvent } from "react";
import { Plus } from "@phosphor-icons/react";
import { PanelFrame } from "../components/PanelFrame";
import { ExternalLink, StaleNotice, panelState } from "../components/bits";
import { ApiError, apiSend } from "../lib/api";
import { relTime } from "../lib/time";
import { useResource } from "../lib/useResource";
import type { Issue } from "../../../shared/types";

// These mirror the server's limits (server/routes/issues.ts).
export const TITLE_MAX_CHARS = 200;
export const BODY_MAX_CHARS = 60_000;

type Field = "title" | "body" | "form";

function validate(title: string, body: string): { field: Field; message: string } | null {
  const t = title.trim();
  if (t.length < 1) return { field: "title", message: "Enter a title." };
  if (t.length > TITLE_MAX_CHARS) return { field: "title", message: `Title must be at most ${TITLE_MAX_CHARS} characters.` };
  if (t.includes("\0")) return { field: "title", message: "Title must not contain NUL characters." };
  if (body.length > BODY_MAX_CHARS) return { field: "body", message: `Details must be at most ${BODY_MAX_CHARS} characters.` };
  if (body.includes("\0")) return { field: "body", message: "Details must not contain NUL characters." };
  return null;
}

/** Which input a server validation message is about. */
function fieldOf(message: string): Field {
  if (/^title\b/i.test(message)) return "title";
  if (/^body\b/i.test(message)) return "body";
  return "form";
}

function NewIssueForm({ onDone, onCancel }: { onDone: (url: string) => void; onCancel: () => void }) {
  const ids = { title: useId(), body: useId(), titleErr: useId(), bodyErr: useId() };
  const titleRef = useRef<HTMLInputElement>(null);
  const bodyRef = useRef<HTMLTextAreaElement>(null);
  const [title, setTitle] = useState("");
  const [body, setBody] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<{ field: Field; message: string } | null>(null);

  const fail = (e: { field: Field; message: string }) => {
    setError(e);
    (e.field === "body" ? bodyRef : titleRef).current?.focus();
  };

  const submit = async (ev: FormEvent) => {
    ev.preventDefault();
    if (pending) return;
    const invalid = validate(title, body);
    if (invalid) return fail(invalid);
    setError(null);
    setPending(true);
    try {
      const res = await apiSend<{ url: string }>("POST", "/api/issues", { title: title.trim(), body });
      onDone(res.url);
    } catch (err) {
      const message = err instanceof ApiError ? err.message : "Could not create the issue.";
      setPending(false);
      fail({ field: err instanceof ApiError && err.status === 400 ? fieldOf(message) : "form", message });
    }
  };

  return (
    <form className="form" onSubmit={submit} aria-label="New issue" noValidate>
      <div className="field">
        <label htmlFor={ids.title}>Title</label>
        <input
          id={ids.title} ref={titleRef} value={title} maxLength={TITLE_MAX_CHARS} autoFocus
          onChange={(e) => setTitle(e.target.value)}
          aria-invalid={error?.field === "title"} aria-describedby={error?.field === "title" ? ids.titleErr : undefined}
        />
        {error?.field === "title" && <p id={ids.titleErr} className="field__error">{error.message}</p>}
      </div>
      <div className="field">
        <label htmlFor={ids.body}>Details (optional)</label>
        <textarea
          id={ids.body} ref={bodyRef} value={body} rows={4}
          onChange={(e) => setBody(e.target.value)}
          aria-invalid={error?.field === "body"} aria-describedby={error?.field === "body" ? ids.bodyErr : undefined}
        />
        {error?.field === "body" && <p id={ids.bodyErr} className="field__error">{error.message}</p>}
      </div>
      {error?.field === "form" && <p className="field__error" role="alert">{error.message}</p>}
      <div className="form__actions">
        <button type="submit" className="btn btn--primary" disabled={pending}>{pending ? "Creating" : "Create issue"}</button>
        <button type="button" className="btn" onClick={onCancel}>Cancel</button>
      </div>
    </form>
  );
}

export function Issues() {
  const res = useResource<Issue[]>("issues", "/api/issues");
  const [open, setOpen] = useState(false);
  const [created, setCreated] = useState<string | null>(null);
  const newBtn = useRef<HTMLButtonElement>(null);
  const state = panelState(res);

  const close = () => {
    setOpen(false);
    // The form unmounts, so hand focus back after React has committed.
    queueMicrotask(() => newBtn.current?.focus());
  };

  const actions =
    state === "ready" ? (
      <button type="button" className="btn" ref={newBtn} aria-expanded={open} onClick={() => { setCreated(null); setOpen(!open); }}>
        <Plus size={14} aria-hidden="true" />
        <span>New issue</span>
      </button>
    ) : undefined;

  return (
    <PanelFrame title="Issues" count={res.data?.length} actions={actions} state={state} error={res.error ?? undefined} onRetry={res.reload}>
      <StaleNotice res={res} label="issues" />
      {open && (
        <NewIssueForm
          onCancel={close}
          onDone={(url) => { setCreated(url); close(); res.reload(); }}
        />
      )}
      <div role="status" className={created ? "notice" : "sr-only"}>
        {created && <>Issue created{ /^https?:\/\//.test(created) && <>: <ExternalLink href={created}>open on GitHub</ExternalLink></>}</>}
      </div>
      {res.data && res.data.length === 0 ? (
        <p className="dim">No open issues. Use New issue to file one.</p>
      ) : (
        <ul className="rows" aria-label="Open issues">
          {res.data?.map((i) => (
            <li key={i.number} className="row row--wrap">
              <span className="mono dim">#{i.number}</span>
              <ExternalLink href={i.url} className="row__grow">{i.title}</ExternalLink>
              {i.labels.map((l) => <span key={l} className="chip">{l}</span>)}
              <span className="dim nowrap">{relTime(i.createdAt)}</span>
            </li>
          ))}
        </ul>
      )}
    </PanelFrame>
  );
}
