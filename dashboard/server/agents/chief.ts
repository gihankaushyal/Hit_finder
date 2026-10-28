import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import type { AgentEvent, AgentEventsResponse, AgentState, ChiefSummary, ConversationInfo } from "../../shared/types";
import type { Config } from "../config";
import type { EventBus } from "../events";
import type { Approvals } from "./approvals";
import type { AgentDriver, Session, SessionStart, ToolDef } from "./driver";
import { adoptEntries, autoTitle, blankEntry, cleanTitle, convDir, isEmptyConv, journalPath, loadRegistry, newConvId, pickContinuable, rebuildRegistry, saveRegistry, sweepTempFiles, TITLE_EDIT_MAX, type ConvEntry, type Registry, type WriteMemo } from "./conversations";
import { eventsAfter, openJournal, readJournalFile, type Journal } from "./journal";
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
const NO_ACTIVE_MESSAGE = "There is no active conversation. Start a new one.";
const SAVE_FAILED_MESSAGE = "The change could not be saved. Check the disk and try again.";
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
  let ready = false;
  const memo: WriteMemo = {};
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
    if (!e) throw new AgentError("state", ready ? NO_ACTIVE_MESSAGE : STARTING_MESSAGE);
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
  /**
   * Queue a save. `bump` marks the active conversation as just used (renaming or archiving another one must not).
   * The returned promise rejects if this save failed; the queue itself carries on either way.
   */
  const persist = (bump = true, ...also: string[]): Promise<void> => {
    const run = writes.then(async () => {
      syncEntry(bump);
      await saveRegistry(dir, reg, [...new Set([reg.active, ...also])], memo);
    });
    writes = run.catch((err) => {
      console.error("chief index write failed", (err as NodeJS.ErrnoException).code ?? "error");
    });
    return run;
  };
  /** A save the caller waits for: a failure becomes an error the owner sees instead of a silent `ok`. */
  const save = (bump = true, ...also: string[]): Promise<void> =>
    persist(bump, ...also).catch(() => {
      throw new AgentError("state", SAVE_FAILED_MESSAGE);
    });
  const touch = (patch: Partial<Meta> = {}) => {
    Object.assign(meta, patch);
    persist().catch(() => undefined); // already logged
    notify();
  };
  const j = (): Journal => {
    if (!journal) throw new AgentError("state", ready ? NO_ACTIVE_MESSAGE : STARTING_MESSAGE);
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
  async function activate(id: string, drop: string | null = null): Promise<void> {
    const e = find(id);
    if (!e) throw new AgentError("not-found", NOT_FOUND_MESSAGE);
    let opened: Journal;
    try {
      opened = await openJournal(journalPath(dir, id));
    } catch (err) {
      journal = null;
      reg.active = "";
      if (drop) delete reg.convs[drop];
      // Nothing of the conversation that was active may linger in the live counters.
      Object.assign(meta, { state: "idle", turns: 0, costUsd: null, sessionId: null, error: null });
      throw err;
    }
    reg.active = id;
    if (drop) delete reg.convs[drop];
    meta.turns = e.turns;
    meta.costUsd = e.costUsd;
    meta.sessionId = e.sessionId;
    // No agent-state event here: the bubble re-seeds from /api/chief, and an "idle" event would raise a "finished" toast.
    meta.state = "idle";
    meta.error = null;
    journal = opened;
  }

  /**
   * Stop the session and file the active conversation away. If it is empty, its id is returned: the caller passes it to
   * `activate` (which drops it from the registry in the same step as the switch, so readers never see a half-switched
   * chief) and then to `settle`, which deletes its folder once the registry is saved. The caller holds `serial`.
   */
  async function park(): Promise<string | null> {
    await stopNow();
    await writes;
    syncEntry(false);
    const e = peek();
    return e && isEmptyConv(e) ? e.id : null;
  }
  const discard = async (id: string | null): Promise<void> => {
    if (id) await fs.rm(convDir(dir, id), { recursive: true, force: true });
  };
  /** End of a switch: the new registry is on disk before the dropped conversation's folder goes (and only then). */
  async function settle(dropped: string | null): Promise<void> {
    await save(); // if this fails the folder stays: the next start finds it and clears it as a leftover
    await discard(dropped);
  }

  const activeEvents = async (after: number, limit: number): Promise<AgentEventsResponse> => {
    if (!journal) return { events: [], lastSeq: 0, conversationId: reg.active };
    return { events: await journal.read(after, limit), lastSeq: journal.lastSeq(), conversationId: reg.active };
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
      await sweepTempFiles(dir);
      const loaded = await loadRegistry(dir);
      if (loaded) await adoptEntries(dir, loaded.convs); // a rebuilt registry has adopted them already
      reg = loaded ?? (await rebuildRegistry(dir, new Date().toISOString()));
      if (!peek() || peek()?.readOnly) {
        // The pointer to the active conversation was lost, or names one that cannot be continued: take the newest one that can still be continued.
        const live = pickContinuable(reg.convs);
        const id = live?.id ?? newConvId(Date.now());
        if (!live) reg.convs[id] = blankEntry(id, new Date().toISOString());
        reg.active = id;
      }
      // Empty conversations other than the active one are leftovers of an interrupted switch.
      const leftovers = Object.values(reg.convs).filter((c) => c.id !== reg.active && isEmptyConv(c));
      for (const c of leftovers) delete reg.convs[c.id];
      // No session survives a restart; the conversation is resumed from its id on the next message.
      await activate(reg.active);
      await saveRegistry(dir, reg, Object.keys(reg.convs), memo);
      for (const c of leftovers) await discard(c.id);
      ready = true;
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
        if (!conv.titleManual && conv.title === null) conv.title = autoTitle(text) || null; // all control characters: no title
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
          try {
            await activate(id, dropped);
          } catch (err) {
            delete reg.convs[id]; // never became active: no blank entry is left behind
            throw err;
          }
          await j().append({ kind: "state", state: "idle", detail: "New conversation" });
        } finally {
          await settle(dropped);
        }
        notify();
      }),

    conversations() {
      // The live counters are shown for the active one without writing them into the entry: a read changes nothing.
      return Object.values(reg.convs)
        .map((c) => (c.id === reg.active ? { ...c, turns: meta.turns, costUsd: meta.costUsd } : c))
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
          await activate(id, dropped);
          target.archived = false; // continuing an archived conversation brings it back; only once it is really active
        } finally {
          await settle(dropped);
        }
        notify();
      }),

    patch: (id, change) =>
      serial(async () => {
        const target = find(id);
        if (!target) throw new AgentError("not-found", NOT_FOUND_MESSAGE);
        const title = change.title === undefined ? undefined : cleanTitle(change.title);
        if (title !== undefined && (title.length === 0 || title.length > TITLE_EDIT_MAX)) {
          throw new AgentError("invalid", `The title must be 1 to ${TITLE_EDIT_MAX} characters.`);
        }
        if (change.archived === true && id === reg.active) throw new AgentError("state", "Start a new conversation before archiving this one.");
        const before = { title: target.title, titleManual: target.titleManual, archived: target.archived };
        if (title !== undefined) {
          target.title = title;
          target.titleManual = true;
        }
        if (change.archived !== undefined) target.archived = change.archived;
        try {
          // Housekeeping, not activity: it must not reorder the list by bumping `updatedAt`.
          await save(false, id);
        } catch (err) {
          Object.assign(target, before); // what the owner is told matches what is on disk
          // The entry file is written before the index, so it may already hold the change: put the old one back if the disk allows.
          await persist(false, id).catch(() => undefined);
          throw err;
        }
        notify();
      }),

    async eventsFor(id, after, limit) {
      if (id === reg.active) return activeEvents(after, limit);
      if (!find(id)) throw new AgentError("not-found", NOT_FOUND_MESSAGE);
      const file = journalPath(dir, id);
      const st = await fs.stat(file).catch(() => null);
      const key = st ? `${st.mtimeMs}:${st.size}` : "";
      if (parked?.id !== id || parked.key !== key) parked = { id, key, all: await readJournalFile(file) };
      const all = parked.all;
      return { events: eventsAfter(all, after, limit), lastSeq: all.length > 0 ? all[all.length - 1].seq : 0, conversationId: id };
    },

    events: activeEvents,
  };
}
