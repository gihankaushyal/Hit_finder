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
export interface SyncState {
  labelsEnsured: boolean;
  items: Record<string, SyncStateItem>;
  /**
   * Issues created by a run that stopped before their marker reached the file, keyed by issue number.
   * `hash` identifies the item (title plus body) so the next run adopts the issue instead of creating another.
   */
  pending?: Record<string, { hash: string }>;
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

export interface SyncDeps {
  gh: Gh;
  readFile(): string | null; // kanban file text, null if missing
  writeFile(text: string): void; // atomic: temp file in the same directory, then rename
  loadState(): SyncState;
  saveState(s: SyncState): void;
  sleep(ms: number): Promise<void>;
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
const footerFor = (section: string): string => `\n\n_Synced from phase-05-kanban.md, section: ${section}_`;
const issueBody = (item: K.MdItem): string => (K.itemBody(item) + footerFor(item.section)).slice(0, BODY_MAX_CHARS);
const createHash = (title: string, item: K.MdItem): string => sha(`${title}\0${issueBody(item)}`);
const issueNumberFrom = (out: string): number | null => {
  const m = /\/issues\/(\d+)\s*$/.exec(out.trim());
  return m ? Number(m[1]) : null;
};

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
}
interface AppendOp {
  task: GhTask;
  closed: boolean;
}

// Runs are serialised per process: a second call waits for the first to finish.
let queue: Promise<unknown> = Promise.resolve();

export function runSync(d: SyncDeps, opts: { dryRun: boolean }): Promise<SyncReport> {
  const result = queue.catch(() => undefined).then(() => execute(d, opts.dryRun));
  queue = result;
  return result;
}

async function execute(d: SyncDeps, dryRun: boolean): Promise<SyncReport> {
  const report: SyncReport = { actions: [], conflicts: [], notInFile: [], skipped: [], aborted: false };
  const tasks = await listTasks(d.gh);
  const byNumber = new Map(tasks.map((t) => [t.number, t]));
  const closedNow = new Map(tasks.map((t) => [t.number, t.state === "CLOSED"]));
  const startText = d.readFile();
  const doc = startText === null ? null : K.parseKanban(startText);
  const loaded = d.loadState();
  const state: SyncState = {
    labelsEnsured: loaded.labelsEnsured === true,
    items: { ...(loaded.items ?? {}) },
    pending: { ...(loaded.pending ?? {}) },
  };
  const savedSnapshot = JSON.stringify(state);

  // ---- plan (pure: no gh calls, no writes) ----
  const linkedOps: LinkedOp[] = [];
  const createOps: CreateOp[] = [];
  const linkOps: LinkOp[] = [];
  const appendOps: AppendOp[] = [];
  const linkedItems: K.MdItem[] = [];
  const referenced = new Set<number>();

  if (doc) {
    // A pending link whose issue is already marked in the file is resolved (the run died before saving state).
    for (const item of K.items(doc)) if (item.issue !== null) delete state.pending![String(item.issue)];
    const adopted = new Set<number>();
    for (const item of K.items(doc)) {
      if (item.issue === null) {
        const title = K.itemTitle(item).slice(0, TITLE_MAX_CHARS).trim();
        const body = issueBody(item);
        if (title === "") {
          report.skipped.push({ title: "", section: item.section, reason: "empty title" });
        } else if (title.includes("\0") || body.includes("\0")) {
          report.skipped.push({ title, section: item.section, reason: "contains a NUL character" });
        } else {
          const hash = createHash(title, item);
          const pendingNumbers = Object.entries(state.pending!)
            .filter(([n, p]) => p.hash === hash && !adopted.has(Number(n)))
            .map(([n]) => Number(n))
            .sort((a, b) => a - b);
          if (pendingNumbers.length > 0) {
            const n = pendingNumbers[0];
            adopted.add(n);
            if (byNumber.has(n)) linkOps.push({ item, title, issue: n });
            else {
              report.skipped.push({
                title,
                section: item.section,
                reason: `#${n} was created for it by an interrupted run but is not visible on GitHub yet; not creating a duplicate`,
              });
            }
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
      if (referenced.has(task.number)) continue;
      if (adopted.has(task.number)) continue;
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
  const record = (item: K.MdItem, bodyUpdated: boolean) => {
    const n = item.issue!;
    const closed = closedNow.get(n) ?? false;
    const prev = state.items[String(n)];
    if (item.checked !== closed) return; // unresolved mismatch: keep the previous record so it stays detectable
    state.items[String(n)] = {
      checked: item.checked,
      closed,
      bodyHash: !prev || bodyUpdated ? sha(K.itemBody(item)) : prev.bodyHash,
    };
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
    if (op.kind === "md-check") {
      K.setChecked(op.item, op.checked);
    } else if (op.kind === "close") {
      await d.gh.text(["issue", "close", String(op.issue)]);
      closedNow.set(op.issue, true);
    } else if (op.kind === "reopen") {
      await d.gh.text(["issue", "reopen", String(op.issue)]);
      closedNow.set(op.issue, false);
    } else {
      await d.gh.text(["issue", "edit", String(op.issue), flagArg("body", issueBody(op.item))]);
      bodyUpdated.add(op.issue);
    }
    report.actions.push(toAction(op));
  }
  if (!flush()) return abort(FILE_CHANGED);
  for (const item of linkedItems) record(item, bodyUpdated.has(item.issue!));
  saveIfChanged();

  // Phase A2: adopt issues created by an interrupted run. The marker goes to disk, then state is saved.
  for (const op of linkOps) {
    K.setIssue(op.item, op.issue);
    if (!flush()) return abort(FILE_CHANGED);
    delete state.pending![String(op.issue)];
    report.actions.push({ type: "link", issue: op.issue, title: op.title });
    const wasClosed = closedNow.get(op.issue) ?? false;
    if (op.item.checked === wasClosed) {
      record(op.item, false);
    } else if (op.item.checked) {
      try {
        await d.gh.text(["issue", "close", String(op.issue)]);
        closedNow.set(op.issue, true);
        record(op.item, false);
      } catch (err) {
        console.error(`could not close #${op.issue} after linking it`, err);
        report.conflicts.push(`#${op.issue} was linked but closing it failed; the next sync will retry`);
      }
    } // else: closed on GitHub, box unticked: the next run follows GitHub
    saveIfChanged();
  }

  // Phase B: creates. After each one the marker is on disk and the state saved before the next begins.
  for (let i = 0; i < createOps.length; i++) {
    const op = createOps[i];
    if (i > 0) await d.sleep(CREATE_PAUSE_MS);
    if (!fileUnchanged()) return abort(FILE_CHANGED);
    const args = ["issue", "create", flagArg("title", op.title), flagArg("body", issueBody(op.item))];
    for (const l of op.labels) args.push(flagArg("label", l));
    const n = issueNumberFrom(await d.gh.text(args));
    if (n === null) throw new Error(`could not read the issue number from gh output for "${op.title}"`);
    closedNow.set(n, false);
    // Order: create, save state (pending), write marker, close if checked, save state.
    state.pending![String(n)] = { hash: op.hash };
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
      try {
        await d.gh.text(["issue", "close", String(n)]);
        closedNow.set(n, true);
        state.items[String(n)] = { ...state.items[String(n)], checked: true, closed: true };
        saveIfChanged();
      } catch (err) {
        console.error(`could not close #${n} after creating it`, err);
        report.conflicts.push(`#${n} was created but closing it failed; the next sync will retry`);
      }
    }
  }

  // Phase C: issues that exist on GitHub but not in the file.
  if (appendOps.length > 0) {
    for (const op of appendOps) K.appendToInbox(doc, op.task.title, op.task.number, op.closed);
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
  return report;
}

// ---- real file and state ----

function writeAtomic(target: string, text: string): void {
  const dir = path.dirname(target);
  fs.mkdirSync(dir, { recursive: true });
  const tmp = path.join(dir, `.${path.basename(target)}.${process.pid}.tmp`);
  let mode: number | undefined;
  try {
    mode = fs.statSync(target).mode & 0o777;
  } catch {
    mode = undefined;
  }
  fs.writeFileSync(tmp, text, mode === undefined ? undefined : { mode });
  fs.renameSync(tmp, target);
}

/** Where the sync state lives; its existence marks that the first import has been done. */
export function syncStatePath(config: Pick<Config, "stateDir">): string {
  return path.join(config.stateDir, STATE_FILE);
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
  return {
    gh,
    readFile() {
      try {
        return fs.readFileSync(config.kanbanPath, "utf8");
      } catch (err) {
        if ((err as NodeJS.ErrnoException).code === "ENOENT") return null;
        throw err;
      }
    },
    writeFile(text) {
      writeAtomic(kanbanTarget(), text);
    },
    loadState() {
      try {
        const raw = JSON.parse(fs.readFileSync(statePath, "utf8")) as Partial<SyncState>;
        if (raw && typeof raw === "object" && raw.items && typeof raw.items === "object") {
          const pending = raw.pending && typeof raw.pending === "object" ? raw.pending : {};
          return { labelsEnsured: raw.labelsEnsured === true, items: raw.items, pending };
        }
      } catch {
        // missing or corrupt: start empty
      }
      return { labelsEnsured: false, items: {} };
    },
    saveState(s) {
      writeAtomic(statePath, JSON.stringify(s, null, 2));
    },
    sleep: (ms) => new Promise((r) => setTimeout(r, ms)),
  };
}
