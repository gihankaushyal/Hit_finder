import type { EventBus } from "../events";
import { flagArg, type Gh } from "./gh";
import * as K from "./kanbanMd";
import {
  labelsForSection, listTasks, runSync, type GhTask, type SyncDeps, type SyncReport,
} from "./kanbanSync";
import type { Board, Task, TaskStatus } from "../../shared/types";

/** Pause after the last file event before a sync starts; the sync's own write is also skipped by content. */
export const FILE_DEBOUNCE_MS = 1500;

const STATUSES: TaskStatus[] = ["todo", "in-progress", "blocked", "done"];
const STATUS_PREFIX = "status:";
const KIND_PREFIX = "kind:";
const SYNC_FAILED = "sync failed; see the server log";
const SYNC_STOPPED = "sync stopped because the kanban file changed on disk; it will retry";
export const NOT_IMPORTED_MESSAGE =
  "The kanban board has not been imported yet. Run the first import from a terminal: npm run kanban:sync -- --dry-run, then npm run kanban:sync -- --yes";

export class NotImportedError extends Error {
  constructor() {
    super(NOT_IMPORTED_MESSAGE);
    this.name = "NotImportedError";
  }
}
export class TaskNotFoundError extends Error {
  constructor(n: number) {
    super(`task #${n} not found`);
    this.name = "TaskNotFoundError";
  }
}

export interface KanbanService {
  board(): Promise<Board>;
  /** Coalescing: calls made while a sync runs share one follow-up run. Rejects with NotImportedError before the first import. */
  sync(): Promise<void>;
  addTask(title: string, body?: string): Promise<void>;
  setStatus(n: number, s: TaskStatus): Promise<void>;
  /** The kanban file changed on disk (debounced; ignored before the import or when it is our own write). */
  fileChanged(): void;
  /** Periodic trigger; ignored before the import. */
  poll(): void;
  dispose(): void;
}

export interface KanbanServiceDeps {
  gh: Gh;
  sync: SyncDeps;
  bus: EventBus;
  /** True once the sync state file exists, i.e. the user has done the first import by hand. */
  imported: () => boolean;
  debounceMs?: number;
}

function statusOf(t: GhTask): TaskStatus {
  if (t.state === "CLOSED") return "done";
  const label = t.labels.find((l) => l.startsWith(STATUS_PREFIX));
  const s = label?.slice(STATUS_PREFIX.length) as TaskStatus | undefined;
  return s && STATUSES.includes(s) && s !== "done" ? s : "todo";
}

const emptyColumns = (): Record<TaskStatus, Task[]> => ({ todo: [], "in-progress": [], blocked: [], done: [] });

export function createKanbanService(d: KanbanServiceDeps): KanbanService {
  const debounceMs = d.debounceMs ?? FILE_DEBOUNCE_MS;
  let tail: Promise<unknown> = Promise.resolve();
  let queuedSync: Promise<void> | null = null;
  let lastReport: SyncReport | null = null;
  let lastSyncAt: string | null = null;
  let syncError: string | null = null;
  let lastWritten: string | null = null;
  let timer: ReturnType<typeof setTimeout> | null = null;

  // Every write path (HTTP mutation, watcher, poll, manual sync) goes through this one queue.
  const exclusive = <T>(fn: () => Promise<T>): Promise<T> => {
    const run = tail.catch(() => undefined).then(fn);
    tail = run;
    return run;
  };
  const requireImported = (): void => {
    if (!d.imported()) throw new NotImportedError();
  };
  const invalidate = (): void => d.bus.emit({ type: "invalidate", resource: "kanban" });

  // Remember what the sync wrote so the file watcher can tell its own write from a user edit.
  const syncDeps: SyncDeps = {
    ...d.sync,
    writeFile(text) {
      lastWritten = text;
      d.sync.writeFile(text);
    },
  };

  async function doSync(): Promise<void> {
    try {
      const report = await runSync(syncDeps, { dryRun: false });
      lastReport = report;
      lastSyncAt = new Date().toISOString();
      syncError = report.aborted ? SYNC_STOPPED : null;
    } catch (err) {
      console.error("kanban sync failed:", err);
      syncError = SYNC_FAILED;
    }
  }

  function sync(): Promise<void> {
    if (!d.imported()) return Promise.reject(new NotImportedError());
    if (queuedSync) return queuedSync;
    const run = exclusive(async () => {
      queuedSync = null; // later triggers now queue a fresh follow-up
      await doSync();
      invalidate();
    });
    queuedSync = run;
    return run;
  }

  function boardFromMarkdown(): Board {
    const columns = emptyColumns();
    const text = d.sync.readFile();
    if (text !== null) {
      K.items(K.parseKanban(text)).forEach((item, i) => {
        const status: TaskStatus = item.checked ? "done" : labelsForSection(item.section).includes("status:in-progress") ? "in-progress" : "todo";
        const kind = labelsForSection(item.section).find((l) => l.startsWith(KIND_PREFIX));
        columns[status].push({
          number: item.issue ?? -(i + 1), // not yet on GitHub: unique negative id
          title: K.itemTitle(item),
          status,
          kind: kind ? kind.slice(KIND_PREFIX.length) : null,
          url: "",
          updatedAt: "",
          inFile: true,
        });
      });
    }
    return { columns, conflicts: [], lastSyncAt: null, syncError: null, imported: false };
  }

  async function board(): Promise<Board> {
    if (!d.imported()) return boardFromMarkdown();
    const tasks = await listTasks(d.gh);
    const notInFile = new Set(lastReport?.notInFile ?? []);
    const columns = emptyColumns();
    for (const t of tasks) {
      const status = statusOf(t);
      const kind = t.labels.find((l) => l.startsWith(KIND_PREFIX));
      columns[status].push({
        number: t.number,
        title: t.title,
        status,
        kind: kind ? kind.slice(KIND_PREFIX.length) : null,
        url: t.url,
        updatedAt: t.updatedAt,
        inFile: !notInFile.has(t.number),
      });
    }
    for (const s of STATUSES) {
      columns[s].sort(s === "done" ? (a, b) => (a.updatedAt < b.updatedAt ? 1 : a.updatedAt > b.updatedAt ? -1 : 0) : (a, b) => a.number - b.number);
    }
    return { columns, conflicts: lastReport?.conflicts ?? [], lastSyncAt, syncError, imported: true };
  }

  async function mutate(fn: () => Promise<void>): Promise<void> {
    requireImported();
    await exclusive(async () => {
      await fn();
      await doSync();
    });
    invalidate();
  }

  const fireAndLog = (what: string): void => {
    sync().catch((err) => {
      if (!(err instanceof NotImportedError)) console.error(`${what} sync failed:`, err);
    });
  };

  return {
    board,
    sync,
    addTask: (title, body) =>
      mutate(async () => {
        await d.gh.text(["issue", "create", flagArg("title", title), flagArg("body", body ?? ""), flagArg("label", "task"), flagArg("label", "status:todo")]);
      }),
    setStatus: (n, s) =>
      mutate(async () => {
        const task = (await listTasks(d.gh)).find((t) => t.number === n);
        if (!task) throw new TaskNotFoundError(n);
        if (s === "done") {
          if (task.state !== "CLOSED") await d.gh.text(["issue", "close", String(n)]);
          return;
        }
        if (task.state === "CLOSED") await d.gh.text(["issue", "reopen", String(n)]);
        const target = `${STATUS_PREFIX}${s}`;
        const stale = task.labels.filter((l) => l.startsWith(STATUS_PREFIX) && l !== target);
        const needsAdd = !task.labels.includes(target);
        if (needsAdd || stale.length > 0) {
          const args = ["issue", "edit", String(n)];
          if (needsAdd) args.push(flagArg("add-label", target));
          for (const l of stale) args.push(flagArg("remove-label", l));
          await d.gh.text(args);
        }
      }),
    fileChanged() {
      if (timer) clearTimeout(timer);
      timer = setTimeout(() => {
        timer = null;
        if (!d.imported()) return;
        let current: string | null;
        try {
          current = d.sync.readFile();
        } catch {
          return;
        }
        if (current !== null && current === lastWritten) return; // our own write
        fireAndLog("file-triggered");
      }, debounceMs);
      timer.unref?.();
    },
    poll() {
      if (d.imported()) fireAndLog("poll");
    },
    dispose() {
      if (timer) clearTimeout(timer);
      timer = null;
    },
  };
}
