import crypto from "node:crypto";
import { spawn as nodeSpawn } from "node:child_process";
import path from "node:path";
import type { Readable, Writable } from "node:stream";
import { fileURLToPath } from "node:url";
import type { Config } from "../config";

export type TerminalErrorCode = "disabled" | "limit" | "invalid" | "not-found" | "state";

/** Messages are written to be shown to the user: they name no paths and carry no process output. */
export class TerminalError extends Error {
  constructor(
    readonly code: TerminalErrorCode,
    message: string,
  ) {
    super(message);
    this.name = "TerminalError";
  }
}

const BRIDGE = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "ptyBridge.py");
const TICKET_MS = 30_000;
const KILL_AFTER_MS = 2_000;
const MAX_INPUT_BYTES = 64 * 1024;
const MIN_COLS = 2;
const MAX_COLS = 500;
const MIN_ROWS = 1;
const MAX_ROWS = 200;
/** The only variables the shell inherits; your login profile sets up the rest, exactly as it does over ssh. */
const ENV_ALLOWED = /^(PATH|HOME|USER|LOGNAME|SHELL|LANG|LANGUAGE|LC_[A-Z_]+|TZ|TMPDIR)$/;

export interface ChildLike {
  stdin: Writable;
  stdout: Readable;
  pid?: number;
  once(event: "exit", listener: (code: number | null, signal: string | null) => void): unknown;
  once(event: "error", listener: (err: Error) => void): unknown;
}
export type SpawnFn = (cmd: string, args: string[], opts: { env: Record<string, string | undefined>; cwd: string }) => ChildLike;

export interface TerminalDeps {
  cfg: Config;
  env: Record<string, string | undefined>;
  spawn?: SpawnFn;
  kill?: (pid: number, signal: NodeJS.Signals) => void;
  ticketMs?: number;
  killAfterMs?: number;
}

export interface Terminal {
  readonly enabled: boolean;
  /** Reserves a stream and returns its id; nothing runs until `attach`. */
  open(cols: number, rows: number): { id: string };
  /** Starts the shell and returns its output. Once per id. */
  attach(id: string): { output: Readable };
  input(id: string, data: string): void;
  resize(id: string, cols: number, rows: number): void;
  close(id: string): Promise<void>;
  closeAll(timeoutMs: number): Promise<void>;
  count(): number;
}

/** The shell gets a short list of ordinary variables and a terminal type; nothing from the dashboard's own environment. */
export function terminalEnv(src: Record<string, string | undefined>): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(src)) if (v !== undefined && ENV_ALLOWED.test(k)) out[k] = v;
  out.TERM = "xterm-256color";
  out.COLORTERM = "truecolor";
  return out;
}

const frame = (type: "d" | "r", payload: Buffer): Buffer => {
  const head = Buffer.alloc(5);
  head.write(type, 0, "latin1");
  head.writeUInt32BE(payload.length, 1);
  return Buffer.concat([head, payload]);
};

function checkSize(cols: number, rows: number): void {
  if (!Number.isInteger(cols) || !Number.isInteger(rows) || cols < MIN_COLS || cols > MAX_COLS || rows < MIN_ROWS || rows > MAX_ROWS) {
    throw new TerminalError("invalid", `Terminal size must be ${MIN_COLS} to ${MAX_COLS} columns and ${MIN_ROWS} to ${MAX_ROWS} rows.`);
  }
}

interface Entry {
  id: string;
  cols: number;
  rows: number;
  createdAt: number;
  child: ChildLike | null;
  exited: Promise<void> | null;
}

export function createTerminal(deps: TerminalDeps): Terminal {
  const { cfg } = deps;
  const ticketMs = deps.ticketMs ?? TICKET_MS;
  const killAfterMs = deps.killAfterMs ?? KILL_AFTER_MS;
  const doSpawn: SpawnFn =
    deps.spawn ??
    ((cmd, args, opts) => nodeSpawn(cmd, args, { env: opts.env as NodeJS.ProcessEnv, cwd: opts.cwd, detached: true, stdio: ["pipe", "pipe", "ignore"] }));
  const doKill = deps.kill ?? ((pid: number, signal: NodeJS.Signals) => process.kill(pid, signal));
  const entries = new Map<string, Entry>();

  const purge = () => {
    const now = Date.now();
    for (const [id, e] of entries) if (!e.child && now - e.createdAt > ticketMs) entries.delete(id);
  };
  const need = (id: string): Entry => {
    purge();
    const e = entries.get(id);
    if (!e) throw new TerminalError("not-found", "Terminal not found. Open it again.");
    return e;
  };
  const command = (): string[] => {
    if (cfg.terminalCommand) return cfg.terminalCommand;
    const socket = cfg.terminalTmuxSocket ? ["-L", cfg.terminalTmuxSocket] : [];
    return ["tmux", ...socket, "new-session", "-A", "-s", cfg.terminalSession, "-c", cfg.repoRoot, ";", "set-option", "status", "off"];
  };

  async function close(id: string): Promise<void> {
    const e = entries.get(id);
    if (!e) return;
    entries.delete(id);
    const child = e.child;
    if (!child) return;
    child.stdin.end();
    const signal = (s: NodeJS.Signals) => {
      try {
        if (child.pid !== undefined) doKill(-child.pid, s); // the helper's whole process group
      } catch {
        /* already gone */
      }
    };
    signal("SIGTERM");
    const gone = await Promise.race([e.exited as Promise<void>, new Promise<"late">((res) => setTimeout(() => res("late"), killAfterMs))]);
    if (gone === "late") signal("SIGKILL");
  }

  return {
    enabled: cfg.terminalEnabled,

    open(cols, rows) {
      if (!cfg.terminalEnabled) throw new TerminalError("disabled", "The terminal is switched off (DASH_TERMINAL=off).");
      checkSize(cols, rows);
      purge();
      if (entries.size >= cfg.terminalMax) throw new TerminalError("limit", `${cfg.terminalMax} terminal streams are already open. Close one first.`);
      const id = crypto.randomUUID();
      entries.set(id, { id, cols, rows, createdAt: Date.now(), child: null, exited: null });
      return { id };
    },

    attach(id) {
      const e = need(id);
      if (e.child) throw new TerminalError("state", "This terminal stream is already attached.");
      const args = [BRIDGE, "--cols", String(e.cols), "--rows", String(e.rows), "--", ...command()];
      const child = doSpawn(cfg.terminalPython, args, { env: terminalEnv(deps.env), cwd: cfg.repoRoot });
      e.child = child;
      e.exited = new Promise<void>((res) => {
        const gone = () => {
          if (entries.get(id) === e) entries.delete(id);
          res();
        };
        child.once("exit", gone);
        // A helper that cannot start (missing interpreter) reports an error and never exits. Unhandled, that would
        // crash the whole dashboard; instead the stream ends and the slot is freed.
        child.once("error", (err) => {
          console.error("terminal helper failed to start:", (err as NodeJS.ErrnoException).code ?? "error");
          child.stdout.destroy();
          child.stdin.destroy();
          gone();
        });
      });
      child.stdin.on("error", () => undefined); // a helper that has gone away must not crash the server
      return { output: child.stdout };
    },

    input(id, data) {
      const e = need(id);
      if (!e.child) throw new TerminalError("state", "The terminal stream is not attached yet.");
      const bytes = Buffer.from(data, "utf8");
      if (bytes.length === 0 || bytes.length > MAX_INPUT_BYTES) throw new TerminalError("invalid", `Input must be 1 to ${MAX_INPUT_BYTES} bytes.`);
      e.child.stdin.write(frame("d", bytes));
    },

    resize(id, cols, rows) {
      const e = need(id);
      checkSize(cols, rows);
      if (!e.child) throw new TerminalError("state", "The terminal stream is not attached yet.");
      const payload = Buffer.alloc(4);
      payload.writeUInt16BE(cols, 0);
      payload.writeUInt16BE(rows, 2);
      e.child.stdin.write(frame("r", payload));
    },

    close,

    async closeAll(timeoutMs) {
      const all = Promise.all([...entries.keys()].map((id) => close(id)));
      await Promise.race([all, new Promise((res) => setTimeout(res, timeoutMs))]);
    },

    count() {
      purge();
      return entries.size;
    },
  };
}
