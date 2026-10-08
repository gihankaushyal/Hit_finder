import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import type { Config } from "../config";
import { flagArg, type Gh } from "./gh";
import * as K from "./kanbanMd";

export interface GhTask {
  number: number;
  title: string;
  state: "OPEN" | "CLOSED";
  labels: string[];
  body: string;
  url: string;
  updatedAt: string;
}
export interface SyncStateItem {
  checked: boolean;
  closed: boolean;
  bodyHash: string;
}
export interface PendingLink {
  /** Identifies the item (title plus body) the issue was created for. */
  hash: string;
  /** When the issue was created (ms since epoch); used to give up waiting for it to appear in the list. */
  at?: number;
}
export interface SyncState {
  labelsEnsured: boolean;
  items: Record<string, SyncStateItem>;
  /** Issues created by a run that stopped before their marker reached the file, keyed by issue number. */
  pending?: Record<string, PendingLink>;
  /** ISO time at which a real run first finished a complete import. Gates every background trigger. */
  importCompletedAt?: string;
}
export type SyncAction =
  | { type: "create"; title: string; labels: string[]; closed: boolean }
  | { type: "link"; issue: number; title: string } // adopt an issue created by an interrupted run
  | { type: "close"; issue: number }
  | { type: "reopen"; issue: number }
  | { type: "update-body"; issue: number }
  | { type: "md-check"; issue: number; checked: boolean }
  | { type: "md-append"; issue: number; title: string };
export interface SyncSkipped {
  title: string;
  section: string;
  reason: string;
}
export interface SyncReport {
  actions: SyncAction[]; // dry run: what a real run would do; real run: what it did
  conflicts: string[];
  notInFile: number[];
  skipped: SyncSkipped[]; // items that were deliberately not created
  aborted: boolean; // true when the run stopped early (the file changed on disk)
}
export interface SyncOptions {
  dryRun: boolean;
  /** Rebuild the state records from the current file and GitHub, flipping and closing nothing. */
  rebuildState?: boolean;
}

export interface SyncDeps {
  gh: Gh;
  readFile(): string | null; // kanban file text, null if missing; throws on invalid UTF-8
  writeFile(text: string): void; // atomic: temp file in the same directory, then rename
  loadState(): SyncState; // missing file: fresh state; unreadable or wrong shape: throws SyncStateError
  saveState(s: SyncState): void;
  sleep(ms: number): Promise<void>;
  /** Exclusive cross-process lock for a real run; throws SyncLockedError when another sync holds it. */
  acquireLock?(): () => void;
  /** Clock in ms since epoch (injectable for tests). */
  now?(): number;
}

/** Another sync (CLI or server) is running; this run did not start. */
export class SyncLockedError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SyncLockedError";
  }
}
/** The sync state file is unreadable, malformed, or inconsistent with the file; nothing was changed. */
export class SyncStateError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SyncStateError";
  }
}

export const TASK_LABELS: { name: string; color: string }[] = [
  { name: "task", color: "0e8a16" },
  { name: "status:todo", color: "c5def5" },
  { name: "status:in-progress", color: "fbca04" },
  { name: "status:blocked", color: "d93f0b" },
  { name: "kind:decision", color: "5319e7" },
  { name: "kind:explore", color: "1d76db" },
  { name: "kind:awaiting-merge", color: "bfdadc" },
  { name: "kind:tracked", color: "ededed" },
];

const TASK_LABEL = "task";
const LIST_LIMIT = "1000";
const LIST_FIELDS = "number,title,state,labels,body,url,updatedAt";
const TITLE_MAX_CHARS = 256; // GitHub's issue title limit
const BODY_MAX_CHARS = 60_000; // GitHub's limit is 65,536
const CREATE_PAUSE_MS = 1000;
const STATE_FILE = "kanban-sync.json";
const LOCK_FILE = "kanban-sync.lock";
const BACKUP_SUFFIX = ".pre-sync.bak";
/** A pending issue that is still absent from the list after this long is reported as a conflict. */
export const PENDING_MAX_AGE_MS = 10 * 60 * 1000;
/** An unparsable lock file younger than this may still be being written by its owner. */
const LOCK_UNPARSABLE_GRACE_MS = 10_000;
const ZERO_WIDTH_JOINER = "\u200D";
/** gh failures that affect every item, so the run stops instead of isolating them. */
const GLOBAL_GH_FAILURE =
  /auth|credential|token|log ?in|\b40[13]\b|rate limit|abuse|could not resolve|network|connection|timed? ?out|ENOTFOUND|ECONN/i;
const FILE_CHANGED = "the kanban file changed on disk during the sync; stopped without overwriting it";

export function labelsForSection(section: string): string[] {
  const labels = [TASK_LABEL, section.startsWith("In progress") ? "status:in-progress" : "status:todo"];
  if (section.startsWith("Inbox")) return labels;
  if (section.startsWith("Awaiting merge")) labels.push("kind:awaiting-merge");
  else if (section.startsWith("Open decisions")) labels.push("kind:decision");
  else if (section.startsWith("Explorable objectives")) labels.push("kind:explore");
  else labels.push("kind:tracked");
  return labels;
}

export async function listTasks(gh: Gh): Promise<GhTask[]> {
  const raw = await gh.json<(Omit<GhTask, "labels"> & { labels: { name: string }[] })[]>([
    "issue", "list", "--label", TASK_LABEL, "--state", "all", "--limit", LIST_LIMIT, "--json", LIST_FIELDS,
  ]);
  return raw.map((t) => ({ ...t, labels: t.labels.map((l) => l.name) }));
}


const sha = (s: string): string => crypto.createHash("sha256").update(s).digest("hex");
/** Stops `@name` text from notifying a real GitHub user. Idempotent. */
export function escapeMentions(s: string): string {
  return s.replace(/@(?!‍)/g, `@${ZERO_WIDTH_JOINER}`);
}
const unescapeMentions = (s: string): string => s.replaceAll(`@${ZERO_WIDTH_JOINER}`, "@");
/** Line endings and trailing whitespace are not significant when matching a body GitHub returned. */
const normBody = (s: string): string => s.replace(/\r\n/g, "\n").trimEnd();
const footerFor = (section: string): string => `\n\n_Synced from phase-05-kanban.md, section: ${section}_`;
const issueBody = (item: K.MdItem): string => escapeMentions(K.itemBody(item) + footerFor(item.section)).slice(0, BODY_MAX_CHARS);
const issueTitle = (item: K.MdItem): string => escapeMentions(K.itemTitle(item)).slice(0, TITLE_MAX_CHARS).trim();
const createHash = (title: string, item: K.MdItem): string => sha(`${title}\0${issueBody(item)}`);
const issueNumberFrom = (out: string): number | null => {
  const m = /\/issues\/(\d+)\s*$/.exec(out.trim());
  return m ? Number(m[1]) : null;
};
const isGlobalFailure = (err: unknown): boolean => GLOBAL_GH_FAILURE.test(err instanceof Error ? err.message : String(err));

type LinkedOp =
  | { kind: "md-check"; item: K.MdItem; issue: number; checked: boolean }
  | { kind: "close" | "reopen" | "update-body"; item: K.MdItem; issue: number };
interface CreateOp {
  item: K.MdItem;
  title: string;
  labels: string[];
  hash: string;
}
interface LinkOp {
  item: K.MdItem;
  title: string;
  issue: number;
  hash: string;
}
interface AppendOp {
  task: GhTask;
  closed: boolean;
}

// Runs are serialised per process: a second call waits for the first to finish.
let queue: Promise<unknown> = Promise.resolve();

export function runSync(d: SyncDeps, opts: SyncOptions): Promise<SyncReport> {
  const result = queue.catch(() => undefined).then(async () => {
    const release = !opts.dryRun && d.acquireLock ? d.acquireLock() : null;
    try {
      return await execute(d, opts);
    } finally {
      release?.();
    }
  });
  queue = result;
  return result;
}

async function execute(d: SyncDeps, opts: SyncOptions): Promise<SyncReport> {
  const dryRun = opts.dryRun;
  const rebuild = opts.rebuildState === true;
  const nowMs = (): number => (d.now ?? Date.now)();
  const report: SyncReport = { actions: [], conflicts: [], notInFile: [], skipped: [], aborted: false };
  // State and file first: a corrupt state file or invalid UTF-8 stops the run before any gh call.
  const loaded = d.loadState();
  const startText = d.readFile();
  const tasks = await listTasks(d.gh);
  const byNumber = new Map(tasks.map((t) => [t.number, t]));
  const closedNow = new Map(tasks.map((t) => [t.number, t.state === "CLOSED"]));
  const doc = startText === null ? null : K.parseKanban(startText);
  const state: SyncState = {
    labelsEnsured: loaded.labelsEnsured === true,
    items: { ...(loaded.items ?? {}) },
    pending: { ...(loaded.pending ?? {}) },
    ...(loaded.importCompletedAt ? { importCompletedAt: loaded.importCompletedAt } : {}),
  };
  const savedSnapshot = JSON.stringify(state);
  const hadRecords = Object.keys(state.items).length > 0 || Object.keys(state.pending!).length > 0;

  // ---- plan (pure: no gh calls, no writes) ----
  const linkedOps: LinkedOp[] = [];
  const createOps: CreateOp[] = [];
  const linkOps: LinkOp[] = [];
  const appendOps: AppendOp[] = [];
  const linkedItems: K.MdItem[] = [];
  const referenced = new Set<number>();
  let incomplete = false; // something that a first import must still do remains undone

  if (doc) {
    const allItems = K.items(doc);
    const markered = allItems.filter((it) => it.issue !== null);
    if (markered.length > 0 && !hadRecords && !rebuild) {
      throw new SyncStateError(
        "the kanban file has issue markers but the sync state has no records, which usually means the state file was lost; " +
          "nothing was changed. If that is expected, run again with --rebuild-state to rebuild the records from the file and GitHub without flipping or closing anything",
      );
    }
    // A pending link whose issue is already marked in the file is resolved (the run died before saving state).
    // If that issue is closed behind an unticked box and has no record yet, record the disagreement so the
    // next step reports it instead of ticking the box.
    for (const item of markered) {
      const key = String(item.issue);
      if (state.pending![key] && !state.items[key] && !item.checked && byNumber.get(item.issue!)?.state === "CLOSED") {
        state.items[key] = { checked: false, closed: true, bodyHash: sha(K.itemBody(item)) };
      }
      delete state.pending![key];
    }
    const markedNumbers = new Set(markered.map((it) => it.issue!));
    const claimed = new Set<number>(); // issues taken by an unmarked item in this run
    // Stateless adoption only considers issues the sync has no record of. The exception is a file in which
    // every marker is gone while the state has records (a stale editor buffer): then recorded issues are
    // candidates too, and an item is re-linked only on an exact title and body match.
    const allMarkersGone = markered.length === 0 && Object.keys(state.items).length > 0;
    const candidates = tasks
      .filter((t) => !markedNumbers.has(t.number) && (allMarkersGone || !state.items[String(t.number)]))
      .sort((a, b) => a.number - b.number);
    // Ticked items pick first and prefer closed issues; unticked items prefer open ones.
    const adoptable = new Map<K.MdItem, GhTask>();
    const pendingHashes = new Set(Object.values(state.pending!).map((p) => p.hash));
    const taken = new Set<number>();
    for (const item of [...allItems.filter((it) => it.issue === null && it.checked), ...allItems.filter((it) => it.issue === null && !it.checked)]) {
      const title = issueTitle(item);
      const body = issueBody(item);
      if (title === "" || title.includes("\0") || body.includes("\0") || pendingHashes.has(createHash(title, item))) continue;
      const matches = candidates.filter((t) => !taken.has(t.number) && t.title === title && normBody(t.body) === normBody(body));
      const preferred = matches.find((t) => (t.state === "CLOSED") === item.checked) ?? matches[0];
      if (preferred) {
        taken.add(preferred.number);
        adoptable.set(item, preferred);
      }
    }
    for (const item of allItems) {
      if (item.issue === null) {
        const title = issueTitle(item);
        const body = issueBody(item);
        if (title === "") {
          report.skipped.push({ title: "", section: item.section, reason: "empty title" });
        } else if (title.includes("\0") || body.includes("\0")) {
          report.skipped.push({ title, section: item.section, reason: "contains a NUL character" });
        } else {
          const hash = createHash(title, item);
          const pendingNumbers = Object.entries(state.pending!)
            .filter(([n, p]) => p.hash === hash && !claimed.has(Number(n)))
            .map(([n]) => Number(n))
            .sort((a, b) => a - b);
          const pendingNumber = pendingNumbers[0];
          const stateless = adoptable.get(item);
          if (pendingNumber !== undefined && byNumber.has(pendingNumber)) {
            claimed.add(pendingNumber);
            linkOps.push({ item, title, issue: pendingNumber, hash });
          } else if (pendingNumber !== undefined) {
            claimed.add(pendingNumber);
            const age = nowMs() - (state.pending![String(pendingNumber)].at ?? nowMs());
            if (age >= PENDING_MAX_AGE_MS) {
              report.conflicts.push(
                `#${pendingNumber} was created for "${title}" by an interrupted run but has still not appeared on GitHub after ${Math.round(PENDING_MAX_AGE_MS / 60000)} minutes; ` +
                  "not creating it again automatically. Check the repository's issues, then add the marker by hand or remove the entry from the sync state",
              );
            } else {
              incomplete = true;
              report.skipped.push({
                title,
                section: item.section,
                reason: `#${pendingNumber} was created for it by an interrupted run but is not visible on GitHub yet; not creating a duplicate`,
              });
            }
          } else if (stateless) {
            claimed.add(stateless.number);
            linkOps.push({ item, title, issue: stateless.number, hash });
          } else {
            createOps.push({ item, title, labels: labelsForSection(item.section), hash });
          }
        }
        continue;
      }
      const n = item.issue;
      if (referenced.has(n)) {
        report.conflicts.push(`#${n} is referenced by more than one item in the file; ignored the later one`);
        continue;
      }
      referenced.add(n);
      const task = byNumber.get(n);
      if (!task) {
        report.conflicts.push(`#${n} is referenced in the file but was not found on GitHub`);
        continue;
      }
      linkedItems.push(item);
      const closed = task.state === "CLOSED";
      if (rebuild) {
        // Record reality; flip nothing. A disagreement is left for the user.
        state.items[String(n)] = { checked: item.checked, closed, bodyHash: sha(K.itemBody(item)) };
      }
      const last = state.items[String(n)];
      if (!last && item.checked && !closed) {
        // Marker present, no record: an import that stopped before closing the issue. Finish it.
        linkedOps.push({ kind: "close", item, issue: n });
      } else if (item.checked !== closed) {
        const mdChanged = last ? item.checked !== last.checked : false;
        const ghChanged = last ? closed !== last.closed : false;
        if (ghChanged || !last) {
          linkedOps.push({ kind: "md-check", item, issue: n, checked: closed });
          if (mdChanged) report.conflicts.push(`#${n} changed in both places; kept GitHub state`);
        } else if (mdChanged) {
          linkedOps.push({ kind: closed ? "reopen" : "close", item, issue: n });
        } else {
          report.conflicts.push(`#${n} differs between the file and GitHub but neither changed since the last sync; left alone`);
        }
      }
      if (last && sha(K.itemBody(item)) !== last.bodyHash) linkedOps.push({ kind: "update-body", item, issue: n });
    }
    for (const task of [...tasks].sort((a, b) => a.number - b.number)) {
      if (referenced.has(task.number) || claimed.has(task.number)) continue;
      if (state.items[String(task.number)] || state.pending![String(task.number)]) report.notInFile.push(task.number);
      else appendOps.push({ task, closed: task.state === "CLOSED" });
    }
  }

  const toAction = (op: LinkedOp): SyncAction =>
    op.kind === "md-check" ? { type: "md-check", issue: op.issue, checked: op.checked } : { type: op.kind, issue: op.issue };
  const createAction = (op: CreateOp): SyncAction => ({
    type: "create", title: op.title, labels: op.labels, closed: op.item.checked,
  });
  const appendAction = (op: AppendOp): SyncAction => ({ type: "md-append", issue: op.task.number, title: op.task.title });

  if (dryRun) {
    report.actions.push(
      ...linkedOps.map(toAction),
      ...linkOps.map((op): SyncAction => ({ type: "link", issue: op.issue, title: op.title })),
      ...createOps.map(createAction), ...appendOps.map(appendAction),
    );
    return report;
  }

  // ---- execute ----
  let known = startText; // the file content we last read or wrote
  const saveIfChanged = (() => {
    let last = savedSnapshot;
    return () => {
      const now = JSON.stringify(state);
      if (now !== last) {
        d.saveState(structuredClone(state));
        last = now;
      }
    };
  })();
  const fileUnchanged = (): boolean => d.readFile() === known;
  const abort = (note: string): SyncReport => {
    report.aborted = true;
    report.conflicts.push(note);
    return report;
  };
  /** Writes the document if it differs from the file; false means the file changed under us. */
  const flush = (): boolean => {
    const out = K.serializeKanban(doc!);
    if (out === known) return true;
    if (!fileUnchanged()) return false;
    d.writeFile(out);
    known = out;
    return true;
  };
  /**
   * Runs one gh write for one item. A failure that is specific to the item becomes a conflict and the run
   * goes on; a failure that would hit every item (auth, network, rate limit) is rethrown to abort the run.
   */
  const attempt = async (what: string, fn: () => Promise<void>): Promise<boolean> => {
    try {
      await fn();
      return true;
    } catch (err) {
      if (isGlobalFailure(err)) throw err;
      console.error(`${what} failed:`, err);
      report.conflicts.push(`${what} failed; the next sync will retry`);
      return false;
    }
  };
  const record = (item: K.MdItem, bodyUpdated: boolean) => {
    const n = item.issue!;
    const closed = closedNow.get(n) ?? false;
    const prev = state.items[String(n)];
    const bodyHash = !prev || bodyUpdated ? sha(K.itemBody(item)) : prev.bodyHash;
    if (item.checked !== closed) {
      // Unresolved mismatch: keep the previous checkbox/closed record so it stays detectable, but remember
      // that the body was pushed so the same body is not pushed again on every sync.
      if (prev && bodyUpdated) state.items[String(n)] = { ...prev, bodyHash };
      return;
    }
    state.items[String(n)] = { checked: item.checked, closed, bodyHash };
  };

  if (!state.labelsEnsured) {
    for (const l of TASK_LABELS) await d.gh.text(["label", "create", l.name, flagArg("color", l.color), "--force"]);
    state.labelsEnsured = true;
  }
  saveIfChanged();
  if (!doc) return report;

  // Phase A: linked items. gh close/reopen/edit are idempotent, so a failure here is simply retried next run.
  const bodyUpdated = new Set<number>();
  for (const op of linkedOps) {
    let ok = true;
    if (op.kind === "md-check") {
      K.setChecked(op.item, op.checked);
    } else if (op.kind === "close") {
      ok = await attempt(`closing #${op.issue}`, async () => {
        await d.gh.text(["issue", "close", String(op.issue)]);
        closedNow.set(op.issue, true);
      });
    } else if (op.kind === "reopen") {
      ok = await attempt(`reopening #${op.issue}`, async () => {
        await d.gh.text(["issue", "reopen", String(op.issue)]);
        closedNow.set(op.issue, false);
      });
    } else {
      ok = await attempt(`updating the body of #${op.issue}`, async () => {
        await d.gh.text(["issue", "edit", String(op.issue), flagArg("body", issueBody(op.item))]);
        bodyUpdated.add(op.issue);
      });
    }
    if (ok) report.actions.push(toAction(op));
  }
  if (!flush()) return abort(FILE_CHANGED);
  for (const item of linkedItems) record(item, bodyUpdated.has(item.issue!));
  saveIfChanged();

  // Phase A2: adopt issues that already exist for an unmarked item (an interrupted run, a stripped marker).
  // Same order as a create: pending is saved, then the marker goes to disk, then the item record is saved,
  // and only then is pending cleared. Adoption never ticks or unticks a box.
  for (const op of linkOps) {
    const key = String(op.issue);
    if (!state.pending![key]) state.pending![key] = { hash: op.hash, at: nowMs() };
    saveIfChanged();
    K.setIssue(op.item, op.issue);
    if (!flush()) return abort(FILE_CHANGED);
    const wasClosed = closedNow.get(op.issue) ?? false;
    const bodyHash = sha(K.itemBody(op.item));
    report.actions.push({ type: "link", issue: op.issue, title: op.title });
    let closeAfter = false;
    if (rebuild) {
      state.items[key] = { checked: op.item.checked, closed: wasClosed, bodyHash };
    } else if (op.item.checked === wasClosed) {
      record(op.item, false);
    } else if (!op.item.checked) {
      // Closed on GitHub behind an unticked box: record the disagreement and leave the choice to the user.
      state.items[key] = { checked: false, closed: true, bodyHash };
      report.conflicts.push(`#${op.issue} is closed on GitHub but its item in the file is not ticked; linked them and left both as they are. Tick the box or reopen the issue`);
    } else {
      // Ticked item, open issue: close it. checked:false in the record so a failed close is retried.
      state.items[key] = { checked: false, closed: false, bodyHash };
      closeAfter = true;
    }
    delete state.pending![key];
    saveIfChanged();
    if (closeAfter) {
      const closedOk = await attempt(`closing #${op.issue} after linking it`, async () => {
        await d.gh.text(["issue", "close", String(op.issue)]);
        closedNow.set(op.issue, true);
      });
      if (closedOk) state.items[key] = { checked: true, closed: true, bodyHash };
      saveIfChanged();
    }
  }

  // Phase B: creates. After each one the marker is on disk and the state saved before the next begins.
  for (let i = 0; i < createOps.length; i++) {
    const op = createOps[i];
    if (i > 0) await d.sleep(CREATE_PAUSE_MS);
    if (!fileUnchanged()) return abort(FILE_CHANGED);
    const args = ["issue", "create", flagArg("title", op.title), flagArg("body", issueBody(op.item))];
    for (const l of op.labels) args.push(flagArg("label", l));
    let out = "";
    const created = await attempt(`creating "${op.title}"`, async () => {
      out = await d.gh.text(args);
    });
    if (!created) {
      incomplete = true;
      continue;
    }
    const n = issueNumberFrom(out);
    if (n === null) throw new Error(`could not read the issue number from gh output for "${op.title}"`);
    closedNow.set(n, false);
    // Order: create, save state (pending), write marker, close if checked, save state.
    state.pending![String(n)] = { hash: op.hash, at: nowMs() };
    saveIfChanged();
    K.setIssue(op.item, n);
    if (!flush()) {
      return abort(`#${n} was created on GitHub but the kanban file changed before its marker could be written; ${FILE_CHANGED}`);
    }
    delete state.pending![String(n)];
    // checked:false so that a checked item whose close fails below is closed by the next run.
    state.items[String(n)] = { checked: false, closed: false, bodyHash: sha(K.itemBody(op.item)) };
    saveIfChanged();
    report.actions.push(createAction(op));
    if (op.item.checked) {
      const closedOk = await attempt(`closing #${n} after creating it`, async () => {
        await d.gh.text(["issue", "close", String(n)]);
        closedNow.set(n, true);
      });
      if (closedOk) {
        state.items[String(n)] = { ...state.items[String(n)], checked: true, closed: true };
        saveIfChanged();
      }
    }
  }

  // Phase C: issues that exist on GitHub but not in the file.
  if (appendOps.length > 0) {
    for (const op of appendOps) K.appendToInbox(doc, unescapeMentions(op.task.title), op.task.number, op.closed);
    if (!flush()) return abort(FILE_CHANGED);
    for (const op of appendOps) {
      const item = K.items(doc).find((it) => it.issue === op.task.number);
      if (item) {
        closedNow.set(op.task.number, op.closed);
        record(item, true);
      }
      report.actions.push(appendAction(op));
    }
    saveIfChanged();
  }

  // The import is complete once a real run got to the end with no create left to do.
  if (!incomplete && !state.importCompletedAt) state.importCompletedAt = new Date(nowMs()).toISOString();
  saveIfChanged();
  return report;
}

// ---- real file and state ----

function fsyncDirBestEffort(dir: string): void {
  try {
    const fd = fs.openSync(dir, "r");
    try {
      fs.fsyncSync(fd);
    } finally {
      fs.closeSync(fd);
    }
  } catch {
    // some filesystems cannot fsync a directory
  }
}

/** Temp file in the same directory, original mode restored (umask must not narrow it), fsync, rename. */
function writeAtomic(target: string, text: string): void {
  const dir = path.dirname(target);
  fs.mkdirSync(dir, { recursive: true });
  const tmp = path.join(dir, `.${path.basename(target)}.${process.pid}.tmp`);
  let mode: number | undefined;
  try {
    mode = fs.statSync(target).mode & 0o7777;
  } catch {
    mode = undefined;
  }
  try {
    const fd = fs.openSync(tmp, "w");
    try {
      fs.writeFileSync(fd, text);
      if (mode !== undefined) fs.fchmodSync(fd, mode);
      fs.fsyncSync(fd);
    } finally {
      fs.closeSync(fd);
    }
    fs.renameSync(tmp, target);
    fsyncDirBestEffort(dir);
  } finally {
    fs.rmSync(tmp, { force: true });
  }
}

/** Copies the file to `<name>.pre-sync.bak` once, with the same mode. An existing backup is never replaced. */
function backupOnce(target: string): void {
  const bak = target + BACKUP_SUFFIX;
  if (fs.existsSync(bak) || !fs.existsSync(target)) return;
  const tmp = path.join(path.dirname(bak), `.${path.basename(bak)}.${process.pid}.tmp`);
  try {
    fs.copyFileSync(target, tmp);
    fs.chmodSync(tmp, fs.statSync(target).mode & 0o7777);
    const fd = fs.openSync(tmp, "r");
    try {
      fs.fsyncSync(fd);
    } finally {
      fs.closeSync(fd);
    }
    if (!fs.existsSync(bak)) fs.renameSync(tmp, bak);
  } finally {
    fs.rmSync(tmp, { force: true });
  }
}

/** Where the sync state lives. */
export function syncStatePath(config: Pick<Config, "stateDir">): string {
  return path.join(config.stateDir, STATE_FILE);
}

const isRecord = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);

/** Validates parsed state; throws SyncStateError naming the problem. */
export function parseSyncState(raw: unknown, where: string): SyncState {
  const bad = (why: string): never => {
    throw new SyncStateError(`the sync state file ${where} is not usable (${why}); nothing was changed. Fix or remove that file by hand`);
  };
  if (!isRecord(raw)) return bad("not a JSON object");
  if (!isRecord(raw.items)) return bad("missing items");
  if (raw.labelsEnsured !== undefined && typeof raw.labelsEnsured !== "boolean") return bad("labelsEnsured is not a boolean");
  for (const [n, it] of Object.entries(raw.items)) {
    if (!/^\d+$/.test(n) || !isRecord(it) || typeof it.checked !== "boolean" || typeof it.closed !== "boolean" || typeof it.bodyHash !== "string") {
      return bad(`bad record for item ${n}`);
    }
  }
  if (raw.pending !== undefined) {
    if (!isRecord(raw.pending)) return bad("pending is not an object");
    for (const [n, p] of Object.entries(raw.pending)) {
      if (!/^\d+$/.test(n) || !isRecord(p) || typeof p.hash !== "string" || (p.at !== undefined && typeof p.at !== "number")) {
        return bad(`bad pending entry ${n}`);
      }
    }
  }
  if (raw.importCompletedAt !== undefined && typeof raw.importCompletedAt !== "string") return bad("importCompletedAt is not a string");
  return {
    labelsEnsured: raw.labelsEnsured === true,
    items: raw.items as SyncState["items"],
    ...(raw.pending ? { pending: raw.pending as SyncState["pending"] } : {}),
    ...(raw.importCompletedAt ? { importCompletedAt: raw.importCompletedAt as string } : {}),
  };
}

function readStateFile(statePath: string): SyncState {
  let text: string;
  try {
    text = fs.readFileSync(statePath, "utf8");
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === "ENOENT") return { labelsEnsured: false, items: {} };
    throw new SyncStateError(`the sync state file ${statePath} could not be read; nothing was changed`);
  }
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    throw new SyncStateError(`the sync state file ${statePath} is not valid JSON; nothing was changed. Fix or remove that file by hand`);
  }
  return parseSyncState(raw, statePath);
}

/** True once a real run has completed the first import. Any unreadable state counts as not imported. */
export function isImportCompleted(config: Pick<Config, "stateDir">): boolean {
  try {
    return Boolean(readStateFile(syncStatePath(config)).importCompletedAt);
  } catch {
    return false;
  }
}

function pidAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (err) {
    return (err as NodeJS.ErrnoException).code === "EPERM";
  }
}

function acquireFileLock(stateDir: string): () => void {
  fs.mkdirSync(stateDir, { recursive: true });
  const lockPath = path.join(stateDir, LOCK_FILE);
  const payload = JSON.stringify({ pid: process.pid, startedAt: new Date().toISOString() });
  const held = (who: string): SyncLockedError =>
    new SyncLockedError(`another kanban sync is in progress (${who}); not starting a second one. Lock file: ${lockPath}`);
  for (let tries = 0; tries < 3; tries++) {
    let fd: number;
    try {
      fd = fs.openSync(lockPath, "wx", 0o600);
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code !== "EEXIST") throw err;
      let holder: { pid?: unknown; startedAt?: unknown } | null = null;
      let ageMs = Infinity;
      try {
        ageMs = Date.now() - fs.statSync(lockPath).mtimeMs;
        holder = JSON.parse(fs.readFileSync(lockPath, "utf8")) as { pid?: unknown; startedAt?: unknown };
      } catch (readErr) {
        if ((readErr as NodeJS.ErrnoException).code === "ENOENT") continue; // released meanwhile
      }
      const pid = holder && typeof holder.pid === "number" ? holder.pid : null;
      if (pid !== null ? pidAlive(pid) : ageMs < LOCK_UNPARSABLE_GRACE_MS) {
        throw held(pid !== null ? `pid ${pid}, started ${String(holder?.startedAt ?? "unknown")}` : "lock being created");
      }
      fs.rmSync(lockPath, { force: true }); // stale: its owner is gone
      continue;
    }
    try {
      fs.writeFileSync(fd, payload);
      fs.fsyncSync(fd);
    } catch (err) {
      fs.rmSync(lockPath, { force: true });
      throw err;
    } finally {
      fs.closeSync(fd);
    }
    return () => {
      try {
        if (fs.readFileSync(lockPath, "utf8") === payload) fs.unlinkSync(lockPath);
      } catch {
        // already gone
      }
    };
  }
  throw held("lost a race for the lock");
}

export function fileSyncDeps(config: Config, gh: Gh): SyncDeps {
  const statePath = syncStatePath(config);
  // Write through a symlink instead of replacing it.
  const kanbanTarget = (): string => {
    try {
      return fs.realpathSync(config.kanbanPath);
    } catch {
      return config.kanbanPath;
    }
  };
  const decoder = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true });
  return {
    gh,
    readFile() {
      let buf: Buffer;
      try {
        buf = fs.readFileSync(config.kanbanPath);
      } catch (err) {
        if ((err as NodeJS.ErrnoException).code === "ENOENT") return null;
        throw err;
      }
      try {
        return decoder.decode(buf);
      } catch {
        throw new Error(`the kanban file ${config.kanbanPath} is not valid UTF-8; stopped without changing anything`);
      }
    },
    writeFile(text) {
      const target = kanbanTarget();
      backupOnce(target);
      writeAtomic(target, text);
    },
    loadState: () => readStateFile(statePath),
    saveState(s) {
      writeAtomic(statePath, JSON.stringify(s, null, 2));
    },
    sleep: (ms) => new Promise((r) => setTimeout(r, ms)),
    acquireLock: () => acquireFileLock(config.stateDir),
  };
}
