import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import type { AgentEvent, AgentEventsResponse, AgentState, ChiefSummary, ConversationInfo } from "../../shared/types";
import type { Config } from "../config";
import type { EventBus } from "../events";
import type { Approvals } from "./approvals";
import type { AgentDriver, Session, SessionStart, ToolDef } from "./driver";
import { autoTitle, blankEntry, convDir, journalPath, loadRegistry, newConvId, rebuildRegistry, saveRegistry, TITLE_EDIT_MAX, type ConvEntry, type Registry } from "./conversations";
import { openJournal, readJournalFile, type Journal } from "./journal";
import { chiefOptions } from "./policy";
import { chiefPrompt } from "./rolePrompts";
import { AgentError, MESSAGE_MAX_CHARS } from "./supervisor";

export const CHIEF_ID = "chief";
export const CHIEF_NAME = "Chief of staff";
export const CHIEF_FAILED_MESSAGE = "The chief of staff session ended with an error.";
const NOTIFY_MS = 150;
const STOP_WAIT_MS = 3000;
const INTERRUPT_WAIT_MS = 2000;
const ACTIVE: AgentState[] = ["running", "waiting-approval", "idle"];
const STARTING_MESSAGE = "The chief of staff is still starting. Try again in a moment.";
const NOT_FOUND_MESSAGE = "That conversation does not exist.";
export const READONLY_MESSAGE = "This conversation can no longer be continued. Read it, or continue in a new conversation.";

interface Meta {
  state: AgentState;
  turns: number;
  costUsd: number | null;
  sessionId: string | null;
  error: string | null;
}

export interface Chief {
  init(): Promise<void>;
  summary(): ChiefSummary;
  send(text: string): Promise<void>;
  stop(): Promise<void>;
  /** Park the conversation and start a new one. The parked one stays on disk, with its session id. */
  reset(): Promise<void>;
  events(after: number, limit: number): Promise<AgentEventsResponse>;
  /** Every conversation, newest first. */
  conversations(): ConversationInfo[];
  /** Park the current conversation and make this one active; its session restarts with the next message. */
  resume(id: string): Promise<void>;
  patch(id: string, change: { title?: string; archived?: boolean }): Promise<void>;
  /** A conversation's log, whether or not it is the active one. */
  eventsFor(id: string, after: number, limit: number): Promise<AgentEventsResponse>;
}

export interface ChiefDeps {
  cfg: Config;
  bus: EventBus;
  driver: AgentDriver;
  approvals: Approvals;
  tools: ToolDef[];
  env: Record<string, string | undefined>;
  notifyMs?: number;
  interruptWaitMs?: number;
}

/**
 * One persistent conversation with the chief of staff. The session starts with the first message and
 * is resumed from its saved id after a stop, a crash or a restart. It reads the repository and calls
 * the dashboard tools; every tool that changes anything waits on an approval card.
 */
export function createChief(deps: ChiefDeps): Chief {
  const { cfg, bus, driver, approvals } = deps;
  const dir = path.join(cfg.stateDir, "chief");
  const sandboxDir = path.join(cfg.agentsDir, ".chief");
  const notifyMs = deps.notifyMs ?? NOTIFY_MS;
  const interruptWaitMs = deps.interruptWaitMs ?? INTERRUPT_WAIT_MS;
  const meta: Meta = { state: "idle", turns: 0, costUsd: null, sessionId: null, error: null };
  // `meta` is the live state of the active conversation; `reg` holds every conversation, the active one included.
  let reg: Registry = { v: 1, active: "", convs: {} };
  let journal: Journal | null = null;
  let session: Session | null = null;
  let loop: Promise<void> | null = null;
  let stopping = false;
  let gen = 0;
  let timer: NodeJS.Timeout | null = null;
  let writes: Promise<void> = Promise.resolve();
  let lock: Promise<unknown> = Promise.resolve();
  /** The last parked journal read, reused while the file is unchanged (a panel polls it). */
  let parked: { id: string; key: string; all: AgentEvent[] } | null = null;

  const notify = () => {
    if (timer) return;
    timer = setTimeout(() => {
      timer = null;
      bus.emit({ type: "invalidate", resource: "chat" });
    }, notifyMs);
  };
  /** Own-property lookup: an id such as "constructor" must not match a member of Object.prototype. */
  const find = (id: string): ConvEntry | undefined => (Object.hasOwn(reg.convs, id) ? reg.convs[id] : undefined);
  const peek = (): ConvEntry | undefined => find(reg.active);
  const active = (): ConvEntry => {
    const e = peek();
    if (!e) throw new AgentError("state", STARTING_MESSAGE);
    return e;
  };
  /** Copies the live counters into the active conversation's entry. */
  const syncEntry = (bump: boolean) => {
    const e = peek();
    if (!e) return;
    e.turns = meta.turns;
    e.costUsd = meta.costUsd;
    e.sessionId = meta.sessionId;
    if (bump) e.updatedAt = new Date().toISOString();
  };
  /** `bump` marks the active conversation as just used; renaming or archiving another one must not. */
  const persist = (bump = true): Promise<void> => {
    writes = writes.then(async () => {
      try {
        syncEntry(bump);
        await saveRegistry(dir, reg);
      } catch (err) {
        console.error("chief index write failed", (err as NodeJS.ErrnoException).code ?? "error");
      }
    });
    return writes;
  };
  const touch = (patch: Partial<Meta> = {}) => {
    Object.assign(meta, patch);
    void persist();
    notify();
  };
  const j = (): Journal => {
    if (!journal) throw new AgentError("state", STARTING_MESSAGE);
    return journal;
  };
  const setState = async (state: AgentState, error: string | null = null, detail?: string) => {
    const changed = meta.state !== state || meta.error !== error;
    if (!changed && !detail) return;
    touch({ state, error });
    // A `detail` entry is journalled even when the state is unchanged; the event is only for real changes.
    if (changed) bus.emit({ type: "agent-state", id: CHIEF_ID, name: CHIEF_NAME, state, ...(error ? { error } : {}) });
    await j().append({ kind: "state", state, ...(detail ? { detail } : {}) });
  };
  const serial = <T>(f: () => Promise<T>): Promise<T> => {
    const p = lock.then(f, f);
    lock = p.catch(() => undefined);
    return p;
  };

  /**
   * Make `id` the active conversation: open its journal, then in one step point the registry, the counters and the
   * journal at it, so a reader never sees a half-switched chief. If the journal cannot be opened nothing is half-done:
   * there is no active conversation until the next reset or resume. The caller holds `serial`.
   */
  async function activate(id: string): Promise<void> {
    const e = find(id);
    if (!e) throw new AgentError("not-found", NOT_FOUND_MESSAGE);
    let opened: Journal;
    try {
      opened = await openJournal(journalPath(dir, id));
    } catch (err) {
      journal = null;
      reg.active = "";
      throw err;
    }
    reg.active = id;
    meta.turns = e.turns;
    meta.costUsd = e.costUsd;
    meta.sessionId = e.sessionId;
    // No agent-state event here: the bubble re-seeds from /api/chief, and an "idle" event would raise a "finished" toast.
    meta.state = "idle";
    meta.error = null;
    journal = opened;
  }

  /**
   * Stop the session and file the active conversation away. An empty untitled one is dropped from the registry and its
   * id returned, to be removed from disk with `discard` once the next conversation is active (`active` keeps pointing
   * at it until then, so readers see one consistent conversation; a crash in between is repaired by `init`).
   * The caller holds `serial`.
   */
  async function park(): Promise<string | null> {
    await stopNow();
    await writes;
    syncEntry(false);
    const e = peek();
    if (e && e.turns === 0 && e.sessionId === null && e.title === null) {
      delete reg.convs[e.id];
      return e.id;
    }
    return null;
  }
  const discard = async (id: string | null): Promise<void> => {
    if (id) await fs.rm(convDir(dir, id), { recursive: true, force: true });
  };

  const activeEvents = async (after: number, limit: number): Promise<AgentEventsResponse> => {
    if (!journal) return { events: [], lastSeq: 0 };
    return { events: await journal.read(after, limit), lastSeq: journal.lastSeq() };
  };

  async function startSession(prompt: string): Promise<void> {
    const resume = meta.sessionId ?? undefined;
    const myGen = ++gen;
    const current = () => gen === myGen;
    // A tool call approved a moment before a stop or reset must not still act.
    const guard = (t: ToolDef): ToolDef => ({
      ...t,
      handler: async (a) => (stopping || !current() ? { text: "Cancelled: the chief of staff was stopped.", isError: true } : t.handler(a)),
    });
    const start: SessionStart = {
      prompt: resume ? prompt : `${chiefPrompt(cfg.repoRoot)}\n\nThe owner says:\n${prompt}`,
      resume,
      claudeBin: cfg.claudeBin,
      options: chiefOptions(cfg, { cwd: sandboxDir, env: deps.env }),
      tools: deps.tools.map(guard),
      // Read-only everywhere; the only writable directory is an empty one of its own. The state directory (access token) is hidden.
      confinement: { worktree: sandboxDir, home: os.homedir(), guarded: [], maskDirs: [path.join(cfg.repoRoot, ".secrets"), cfg.stateDir] },
      canUseTool: async (tool, input, ctx) => {
        let approvalId = "";
        const outcome = await approvals.request({ id: CHIEF_ID, name: CHIEF_NAME }, tool, input, {
          reason: ctx.reason,
          signal: ctx.signal,
          onCreated: (a) => {
            approvalId = a.id;
            void setState("waiting-approval").then(() => j().append({ kind: "approval", approvalId, tool, outcome: "requested" }));
          },
        });
        await j().append({ kind: "approval", approvalId, tool, outcome: outcome.cancelled ? "cancelled" : outcome.decision });
        if (!stopping && meta.state === "waiting-approval" && approvals.countFor(CHIEF_ID) === 0) await setState("running");
        return outcome;
      },
    };
    await fs.mkdir(sandboxDir, { recursive: true });
    stopping = false;
    const s = driver.start(start);
    session = s;
    await setState("running");
    loop = (async () => {
      try {
        for await (const item of s.events) {
          if (!current()) continue;
          if ("sessionId" in item) {
            touch({ sessionId: item.sessionId });
            continue;
          }
          await j().append(item.body);
          if (!current()) continue;
          if (item.body.kind === "result") {
            touch({ turns: meta.turns + 1, costUsd: item.body.costUsd === null ? meta.costUsd : Math.max(meta.costUsd ?? 0, item.body.costUsd) });
            if (!stopping) await setState("idle");
          } else notify(); // a streamed message changes nothing that is saved
        }
        if (!stopping && current()) {
          session = null;
          await setState("stopped");
        }
      } catch (err) {
        if (stopping || !current()) return;
        console.error("chief session failed:", err instanceof Error ? err.message : err);
        approvals.cancelAgent(CHIEF_ID);
        await j().append({ kind: "error", message: CHIEF_FAILED_MESSAGE });
        await setState("failed", CHIEF_FAILED_MESSAGE);
        try {
          s.close();
        } catch {
          /* already closed */
        }
        if (session === s) session = null;
      }
    })();
  }

  async function stopNow(): Promise<void> {
    const s = session;
    if (!s) return;
    stopping = true;
    gen++; // whatever this session still yields after this point belongs to no one
    approvals.cancelAgent(CHIEF_ID);
    await setState("stopped");
    try {
      await Promise.race([s.interrupt(), new Promise((res) => setTimeout(res, interruptWaitMs))]);
    } catch {
      /* the session may already be gone */
    }
    s.close();
    if (loop) await Promise.race([loop, new Promise((res) => setTimeout(res, STOP_WAIT_MS))]);
    if (session === s) session = null;
    await writes;
  }

  return {
    async init() {
      await fs.mkdir(dir, { recursive: true });
      reg = (await loadRegistry(dir)) ?? (await rebuildRegistry(dir, new Date().toISOString()));
      if (!peek() || peek()?.readOnly) {
        // The pointer to the active conversation was lost, or names one that cannot be continued: take the newest one that can still be continued.
        const live = Object.values(reg.convs).filter((c) => !c.archived && !c.readOnly).sort((a, b) => b.updatedAt.localeCompare(a.updatedAt))[0];
        const id = live?.id ?? newConvId(Date.now());
        if (!live) reg.convs[id] = blankEntry(id, new Date().toISOString());
        reg.active = id;
      }
      // No session survives a restart; the conversation is resumed from its id on the next message.
      await activate(reg.active);
      await saveRegistry(dir, reg);
    },

    summary: () => ({
      state: meta.state, model: cfg.chiefModel, turns: meta.turns, costUsd: meta.costUsd, sessionId: meta.sessionId, error: meta.error,
      pendingApprovals: approvals.countFor(CHIEF_ID), lastSeq: journal?.lastSeq() ?? 0,
      conversationId: reg.active, title: peek()?.title ?? null, readOnly: peek()?.readOnly ?? null,
    }),

    send(text) {
      if (!text.trim()) return Promise.reject(new AgentError("invalid", "The message cannot be empty."));
      if (text.length > MESSAGE_MAX_CHARS) return Promise.reject(new AgentError("invalid", `Messages are limited to ${MESSAGE_MAX_CHARS} characters.`));
      return serial(async () => {
        const jr = j();
        const conv = active();
        if (conv.readOnly) throw new AgentError("state", READONLY_MESSAGE);
        await jr.append({ kind: "user", text });
        if (!conv.titleManual && conv.title === null) conv.title = autoTitle(text);
        touch(); // a message is activity: it bumps updatedAt
        const live = session;
        if (live && ACTIVE.includes(meta.state)) {
          if (meta.state === "idle") await setState("running");
          // The session may have died while the log was written; if so, fall through and start a new one.
          if (session === live && ACTIVE.includes(meta.state)) {
            live.send(text);
            return;
          }
        }
        await startSession(text);
      });
    },

    stop: () => serial(stopNow),

    reset: () =>
      serial(async () => {
        const dropped = await park();
        try {
          const id = newConvId(Date.now());
          reg.convs[id] = blankEntry(id, new Date().toISOString());
          await activate(id);
          await j().append({ kind: "state", state: "idle", detail: "New conversation" });
        } finally {
          await discard(dropped);
        }
        touch();
        await writes;
      }),

    conversations() {
      syncEntry(false);
      return Object.values(reg.convs)
        .sort((a, b) => b.updatedAt.localeCompare(a.updatedAt))
        .map((c) => ({
          id: c.id, title: c.title, createdAt: c.createdAt, updatedAt: c.updatedAt, turns: c.turns, costUsd: c.costUsd,
          archived: c.archived, readOnly: c.readOnly, active: c.id === reg.active, seededFrom: c.seededFrom,
        }));
    },

    resume: (id) =>
      serial(async () => {
        const target = find(id);
        if (!target) throw new AgentError("not-found", NOT_FOUND_MESSAGE);
        if (id === reg.active) return;
        if (target.readOnly) throw new AgentError("state", READONLY_MESSAGE);
        const dropped = await park();
        try {
          await activate(id);
          target.archived = false; // continuing an archived conversation brings it back; only once it is really active
        } finally {
          await discard(dropped);
        }
        touch();
        await writes;
      }),

    patch: (id, change) =>
      serial(async () => {
        const target = find(id);
        if (!target) throw new AgentError("not-found", NOT_FOUND_MESSAGE);
        const title = change.title === undefined ? undefined : change.title.trim();
        if (title !== undefined && (title.length === 0 || title.length > TITLE_EDIT_MAX)) {
          throw new AgentError("invalid", `The title must be 1 to ${TITLE_EDIT_MAX} characters.`);
        }
        if (change.archived === true && id === reg.active) throw new AgentError("state", "Start a new conversation before archiving this one.");
        if (title !== undefined) {
          target.title = title;
          target.titleManual = true;
        }
        if (change.archived !== undefined) target.archived = change.archived;
        // Housekeeping, not activity: it must not reorder the list by bumping `updatedAt`.
        void persist(false);
        notify();
        await writes;
      }),

    async eventsFor(id, after, limit) {
      if (id === reg.active) return activeEvents(after, limit);
      if (!find(id)) throw new AgentError("not-found", NOT_FOUND_MESSAGE);
      const file = journalPath(dir, id);
      const st = await fs.stat(file).catch(() => null);
      const key = st ? `${st.mtimeMs}:${st.size}` : "";
      if (parked?.id !== id || parked.key !== key) parked = { id, key, all: await readJournalFile(file) };
      const all = parked.all;
      return { events: all.filter((e) => e.seq > after).slice(0, Math.max(0, limit)), lastSeq: all.length > 0 ? all[all.length - 1].seq : 0 };
    },

    events: activeEvents,
  };
}
