import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import type { AgentEvent, AgentEventsResponse, AgentState, AgentSummary } from "../../shared/types";
import type { Config } from "../config";
import type { EventBus } from "../events";
import type { Approvals } from "./approvals";
import type { AgentDriver, Session, SessionStart } from "./driver";
import { openJournal, type Journal } from "./journal";
import { workerOptions } from "./policy";
import { RESUME_PROMPT, workerPrompt } from "./rolePrompts";
import { localRemotePath, validAgentName, type Worktrees } from "./worktrees";

export type AgentErrorCode = "invalid" | "cap" | "exists" | "not-found" | "state";

export class AgentError extends Error {
  constructor(
    readonly code: AgentErrorCode,
    message: string,
  ) {
    super(message);
    this.name = "AgentError";
  }
}

export const TASK_MAX_CHARS = 8000;
export const MESSAGE_MAX_CHARS = 8000;
const NOTIFY_MS = 150;
const STOP_WAIT_MS = 3000;
export const FAILED_MESSAGE = "The agent session ended with an error.";

const ACTIVE: AgentState[] = ["starting", "running", "waiting-approval", "idle"];
type Meta = Omit<AgentSummary, "pendingApprovals" | "lastSeq">;

interface Rec {
  meta: Meta;
  journal: Journal;
  dir: string;
  session: Session | null;
  loop: Promise<void> | null;
  stopping: boolean;
}

export interface Supervisor {
  init(): Promise<void>;
  spawn(input: { name: string; task: string; issue?: number | null }): Promise<AgentSummary>;
  message(id: string, text: string): Promise<void>;
  stop(id: string): Promise<void>;
  resume(id: string): Promise<void>;
  list(): AgentSummary[];
  get(id: string): AgentSummary | undefined;
  events(id: string, after: number, limit: number): Promise<AgentEventsResponse>;
  stopAll(timeoutMs: number): Promise<void>;
}

export interface SupervisorDeps {
  cfg: Config;
  bus: EventBus;
  driver: AgentDriver;
  approvals: Approvals;
  worktrees: Worktrees;
  env: Record<string, string | undefined>;
  /** Delay for coalescing change notifications. */
  notifyMs?: number;
  /** Runs once at init; the real server uses it to kill leftover claude processes. */
  sweep?: () => Promise<void>;
}

async function writeMeta(dir: string, meta: Meta): Promise<void> {
  const tmp = path.join(dir, `meta.json.${process.pid}.tmp`);
  await fs.writeFile(tmp, JSON.stringify(meta, null, 2));
  await fs.rename(tmp, path.join(dir, "meta.json"));
}

export function createSupervisor(deps: SupervisorDeps): Supervisor {
  const { cfg, bus, driver, approvals, worktrees } = deps;
  const root = path.join(cfg.stateDir, "agents");
  const recs = new Map<string, Rec>();
  const notifyMs = deps.notifyMs ?? NOTIFY_MS;
  let timer: NodeJS.Timeout | null = null;

  const notify = () => {
    if (timer) return;
    timer = setTimeout(() => {
      timer = null;
      bus.emit({ type: "invalidate", resource: "agents" });
    }, notifyMs);
  };

  const summary = (r: Rec): AgentSummary => ({ ...r.meta, pendingApprovals: approvals.countFor(r.meta.id), lastSeq: r.journal.lastSeq() });
  const need = (id: string): Rec => {
    const r = recs.get(id);
    if (!r) throw new AgentError("not-found", "Agent not found.");
    return r;
  };

  // One write at a time per agent: a shared temp file would otherwise be renamed away mid-write.
  const writes = new WeakMap<Rec, Promise<void>>();
  const persist = (r: Rec): Promise<void> => {
    const next = (writes.get(r) ?? Promise.resolve()).then(async () => {
      try {
        await writeMeta(r.dir, r.meta);
      } catch (err) {
        console.error("agent meta write failed", (err as NodeJS.ErrnoException).code ?? "error");
      }
    });
    writes.set(r, next);
    return next;
  };

  const touch = (r: Rec, patch: Partial<Meta> = {}) => {
    Object.assign(r.meta, patch, { updatedAt: new Date().toISOString() });
    void persist(r);
    notify();
  };

  const setState = async (r: Rec, state: AgentState, error: string | null = null) => {
    if (r.meta.state === state && r.meta.error === error) return;
    touch(r, { state, error });
    await r.journal.append({ kind: "state", state });
  };

  const activeWorkers = () => [...recs.values()].filter((r) => r.meta.role === "worker" && ACTIVE.includes(r.meta.state)).length;

  async function startSession(r: Rec, prompt: string, resume?: string): Promise<void> {
    const worktree = r.meta.worktree as string;
    const branch = r.meta.branch as string;
    const remote = cfg.agentDriver === "sdk" ? await localRemotePath(cfg.repoRoot) : undefined;
    const start: SessionStart = {
      prompt,
      resume,
      claudeBin: cfg.claudeBin,
      options: workerOptions(cfg, { cwd: worktree, env: deps.env, model: r.meta.model }),
      confinement: {
        worktree,
        gitDir: path.join(cfg.repoRoot, ".git"),
        ownBranch: branch,
        home: os.homedir(),
        remote,
        guarded: [],
        maskDirs: [path.join(cfg.repoRoot, ".secrets")],
      },
      canUseTool: async (tool, input, ctx) => {
        let approvalId = "";
        const outcome = await approvals.request({ id: r.meta.id, name: r.meta.name }, tool, input, {
          reason: ctx.reason,
          signal: ctx.signal,
          onCreated: (a) => {
            approvalId = a.id;
            // The card exists before the state flips, so a reader that sees "waiting-approval" also finds the card.
            void setState(r, "waiting-approval").then(() => r.journal.append({ kind: "approval", approvalId, tool, outcome: "requested" }));
          },
        });
        await r.journal.append({ kind: "approval", approvalId, tool, outcome: outcome.cancelled ? "cancelled" : outcome.decision });
        if (!r.stopping && r.meta.state === "waiting-approval" && approvals.countFor(r.meta.id) === 0) await setState(r, "running");
        return outcome;
      },
    };
    const session = driver.start(start);
    r.session = session;
    await setState(r, "running");
    r.loop = (async () => {
      try {
        for await (const item of session.events) {
          if ("sessionId" in item) {
            touch(r, { sessionId: item.sessionId });
            continue;
          }
          await r.journal.append(item.body);
          if (item.body.kind === "result") {
            touch(r, { turns: r.meta.turns + 1, costUsd: item.body.costUsd === null ? r.meta.costUsd : Math.max(r.meta.costUsd ?? 0, item.body.costUsd) });
            if (!r.stopping) await setState(r, "idle");
          } else {
            touch(r);
          }
        }
        if (!r.stopping) await setState(r, "done");
      } catch (err) {
        if (r.stopping) return;
        console.error(`agent ${r.meta.name} failed:`, err instanceof Error ? err.message : err);
        approvals.cancelAgent(r.meta.id);
        await r.journal.append({ kind: "error", message: FAILED_MESSAGE });
        await setState(r, "failed", FAILED_MESSAGE);
      }
    })();
  }

  async function stop(id: string): Promise<void> {
    const r = need(id);
    if (!ACTIVE.includes(r.meta.state)) return;
    r.stopping = true;
    approvals.cancelAgent(id);
    await setState(r, "stopped");
    const s = r.session;
    if (s) {
      try {
        await s.interrupt();
      } catch {
        /* the session may already be gone */
      }
      s.close();
    }
    if (r.loop) await Promise.race([r.loop, new Promise((res) => setTimeout(res, STOP_WAIT_MS))]);
    r.session = null;
  }

  return {
    async init() {
      await fs.mkdir(root, { recursive: true });
      await deps.sweep?.();
      for (const entry of await fs.readdir(root, { withFileTypes: true })) {
        if (!entry.isDirectory()) continue;
        const dir = path.join(root, entry.name);
        let meta: Meta;
        try {
          meta = JSON.parse(await fs.readFile(path.join(dir, "meta.json"), "utf8")) as Meta;
          if (typeof meta.id !== "string" || !validAgentName(meta.name)) continue;
        } catch {
          continue;
        }
        const journal = await openJournal(path.join(dir, "journal.jsonl"));
        const r: Rec = { meta, journal, dir, session: null, loop: null, stopping: false };
        if (ACTIVE.includes(meta.state)) {
          meta.state = "stopped";
          await journal.append({ kind: "state", state: "stopped", detail: "The dashboard restarted." });
          await persist(r);
        }
        recs.set(meta.id, r);
      }
    },

    async spawn({ name, task, issue = null }) {
      if (!validAgentName(name)) throw new AgentError("invalid", "Agent name must be 2 to 31 characters: lowercase letters, digits and hyphens.");
      if (!task.trim()) throw new AgentError("invalid", "The task cannot be empty.");
      if (task.length > TASK_MAX_CHARS) throw new AgentError("invalid", `The task is limited to ${TASK_MAX_CHARS} characters.`);
      if (issue !== null && (!Number.isInteger(issue) || issue < 1)) throw new AgentError("invalid", "Issue must be a positive whole number.");
      if (recs.has(name)) throw new AgentError("exists", `An agent named "${name}" already exists.`);
      if (activeWorkers() >= cfg.agentMax) throw new AgentError("cap", `${cfg.agentMax} agents are already running. Stop one first.`);
      let wt;
      try {
        wt = await worktrees.create(name);
      } catch (err) {
        throw new AgentError("exists", err instanceof Error ? err.message : "Could not create the worktree.");
      }
      const dir = path.join(root, name);
      const journal = await openJournal(path.join(dir, "journal.jsonl"));
      const now = new Date().toISOString();
      const meta: Meta = {
        id: name, name, role: "worker", task, issue, state: "starting", model: cfg.workerModel, branch: wt.branch, worktree: wt.path,
        createdAt: now, updatedAt: now, lastActivity: null, turns: 0, costUsd: null, sessionId: null, error: null,
      };
      const r: Rec = { meta, journal, dir, session: null, loop: null, stopping: false };
      recs.set(name, r);
      await persist(r);
      await journal.append({ kind: "user", text: task });
      await startSession(r, workerPrompt({ name, branch: wt.branch, issue, task }));
      return summary(r);
    },

    async message(id, text) {
      const r = need(id);
      if (!text.trim()) throw new AgentError("invalid", "The message cannot be empty.");
      if (text.length > MESSAGE_MAX_CHARS) throw new AgentError("invalid", `Messages are limited to ${MESSAGE_MAX_CHARS} characters.`);
      if (!ACTIVE.includes(r.meta.state) || !r.session) throw new AgentError("state", "The agent is not running. Resume it first.");
      await r.journal.append({ kind: "user", text });
      if (r.meta.state === "idle") await setState(r, "running");
      r.session.send(text);
    },

    stop,

    async resume(id) {
      const r = need(id);
      if (ACTIVE.includes(r.meta.state)) throw new AgentError("state", "The agent is already running.");
      if (!r.meta.sessionId) throw new AgentError("state", "This agent has no session to resume.");
      if (activeWorkers() >= cfg.agentMax) throw new AgentError("cap", `${cfg.agentMax} agents are already running. Stop one first.`);
      r.stopping = false;
      await setState(r, "starting");
      await startSession(r, RESUME_PROMPT, r.meta.sessionId);
    },

    list: () => [...recs.values()].map(summary).sort((a, b) => b.createdAt.localeCompare(a.createdAt)),
    get: (id) => (recs.has(id) ? summary(recs.get(id) as Rec) : undefined),

    async events(id, after, limit) {
      const r = need(id);
      const events: AgentEvent[] = await r.journal.read(after, limit);
      return { events, lastSeq: r.journal.lastSeq() };
    },

    async stopAll(timeoutMs) {
      const all = Promise.all([...recs.keys()].map((id) => stop(id).catch(() => undefined)));
      await Promise.race([all, new Promise((res) => setTimeout(res, timeoutMs))]);
    },
  };
}
