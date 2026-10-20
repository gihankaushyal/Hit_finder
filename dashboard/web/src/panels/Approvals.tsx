import { useEffect, useId, useRef, useState } from "react";
import { PanelFrame } from "../components/PanelFrame";
import { panelState } from "../components/bits";
import { ApiError, apiSend } from "../lib/api";
import { relTime } from "../lib/time";
import { AGENTS_REFRESH_MS, useResource } from "../lib/useResource";
import type { Approval, ApprovalDecision, ApprovalsResponse } from "../../../shared/types";

/** More lines than this get a notice, because the box scrolls and the rest would be easy to miss. */
const LONG_COMMAND_LINES = 8;
/** A text argument longer than this (or with line breaks) is shown as real lines instead of one escaped JSON string. */
const LONG_ARGUMENT_CHARS = 200;
const FILE_TOOLS = new Set(["Edit", "Write", "NotebookEdit", "MultiEdit"]);

/** What the agent actually asked for, as text. Never HTML: the content is model-supplied. */
function Request({ a }: { a: Approval }) {
  const input = a.input;
  if (a.tool === "Bash" && typeof input.command === "string") {
    const lines = input.command.split("\n").length;
    return (
      <>
        <pre className="mono approval__code">{input.command}</pre>
        {lines > LONG_COMMAND_LINES && <p className="approval__meta">This command has {lines} lines; scroll the box to read all of it.</p>}
      </>
    );
  }
  if (FILE_TOOLS.has(a.tool) && typeof input.file_path === "string") {
    const { file_path, ...rest } = input;
    return (
      <>
        <p className="mono approval__path">{file_path}</p>
        {Object.keys(rest).length > 0 && <pre className="mono approval__code">{JSON.stringify(rest, null, 2)}</pre>}
      </>
    );
  }
  const worker = a.tool.endsWith("spawn_worker") || a.tool.endsWith("message_worker");
  const longText = Object.values(input).some((v) => typeof v === "string" && (v.includes("\n") || v.length > LONG_ARGUMENT_CHARS));
  return (
    <>
      {worker && <p className="approval__meta">A worker runs repository code with network access and readable credentials. Read the whole task before approving.</p>}
      {longText ? (
        Object.entries(input).map(([k, v]) =>
          typeof v === "string" ? (
            <div key={k}>
              <p className="approval__meta">{k}: {v.length} characters</p>
              <pre className="mono approval__code">{v}</pre>
            </div>
          ) : (
            <pre key={k} className="mono approval__code">{`${k}: ${JSON.stringify(v)}`}</pre>
          ),
        )
      ) : (
        <pre className="mono approval__code">{JSON.stringify(input, null, 2)}</pre>
      )}
    </>
  );
}

function Card({ a, busy, error, onDecide }: { a: Approval; busy: boolean; error: string | null; onDecide: (d: ApprovalDecision) => void }) {
  const titleId = useId();
  return (
    <article className="approval" aria-labelledby={titleId}>
      <h3 id={titleId} className="approval__title">
        <span className="mono">{a.agentName}</span> <span className="dim">wants to use</span> <span className="mono">{a.tool}</span>
      </h3>
      <p className="dim approval__meta">{relTime(a.requestedAt)}{a.reason ? ` · ${a.reason}` : ""}</p>
      <Request a={a} />
      {error && (
        <p className="approval__error" role="alert">{error}</p>
      )}
      <div className="row">
        <button type="button" className="btn btn--primary" data-approve disabled={busy} onClick={() => onDecide("allow")}>Approve</button>
        <button type="button" className="btn" disabled={busy} onClick={() => onDecide("deny")}>Deny</button>
      </div>
    </article>
  );
}

export function Approvals() {
  const res = useResource<ApprovalsResponse>("approvals", "/api/approvals", (d) => d.approvals.length === 0, AGENTS_REFRESH_MS);
  const [busy, setBusy] = useState<ReadonlySet<string>>(new Set());
  const [errors, setErrors] = useState<Readonly<Record<string, string>>>({});
  const sending = useRef(new Set<string>());
  const body = useRef<HTMLDivElement>(null);
  const refocus = useRef(false);
  const list = res.data?.approvals ?? [];

  // After a card is decided and gone, keep keyboard focus in the list instead of dropping it to the page.
  useEffect(() => {
    if (!refocus.current) return;
    refocus.current = false;
    const next = body.current?.querySelector<HTMLElement>("[data-approve]");
    (next ?? body.current)?.focus();
  }, [list.length]);

  async function decide(id: string, decision: ApprovalDecision): Promise<void> {
    if (sending.current.has(id)) return;
    sending.current.add(id);
    setBusy(new Set(sending.current));
    setErrors((e) => ({ ...e, [id]: "" }));
    try {
      await apiSend("POST", `/api/approvals/${encodeURIComponent(id)}/decision`, { decision });
      refocus.current = true;
      res.reload();
    } catch (err) {
      setErrors((e) => ({ ...e, [id]: err instanceof ApiError ? err.message : "Could not send the decision." }));
    } finally {
      sending.current.delete(id);
      setBusy(new Set(sending.current));
    }
  }

  const n = list.length;
  return (
    <div id="approvals">
      <PanelFrame title="Approvals" count={n} state={panelState(res)} error={res.error ?? undefined} onRetry={res.reload}>
        <p className="sr-only" role="status" aria-live="polite">{n === 0 ? "No approvals waiting" : `${n} approval${n === 1 ? "" : "s"} waiting`}</p>
        <div ref={body} tabIndex={-1} className="approvals">
          {n === 0 ? <p className="dim">Nothing is waiting for you.</p> : list.map((a) => (
            <Card key={a.id} a={a} busy={busy.has(a.id)} error={errors[a.id] || null} onDecide={(d) => void decide(a.id, d)} />
          ))}
        </div>
      </PanelFrame>
    </div>
  );
}
