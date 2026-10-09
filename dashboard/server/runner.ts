import { execFile } from "node:child_process";

export interface RunResult { stdout: string; stderr: string; code: number }
export interface RunOptions { cwd?: string; timeoutMs?: number }
export type Runner = (cmd: string, args: string[], opts?: RunOptions) => Promise<RunResult>;

/** Default per-call timeout; long-running callers pass a larger `timeoutMs`. */
export const RUN_TIMEOUT_MS = 60_000;

const MAX_BUFFER_BYTES = 32 * 1024 * 1024;
const SPAWN_FAILURE_CODE = 127;
const TIMEOUT_CODE = 124;

export const execRunner: Runner = (cmd, args, opts) =>
  new Promise((resolve) => {
    const timeoutMs = opts?.timeoutMs ?? RUN_TIMEOUT_MS;
    execFile(
      cmd,
      args,
      { cwd: opts?.cwd, maxBuffer: MAX_BUFFER_BYTES, encoding: "utf8", timeout: timeoutMs, killSignal: "SIGKILL" },
      (error, stdout, stderr) => {
        if (!error) {
          resolve({ stdout, stderr, code: 0 });
          return;
        }
        const e = error as NodeJS.ErrnoException & { code?: number | string; killed?: boolean };
        if (e.killed) {
          const note = `process timed out after ${timeoutMs} ms`;
          resolve({ stdout, stderr: stderr ? `${stderr}\n${note}` : note, code: TIMEOUT_CODE });
          return;
        }
        if (typeof e.code === "number") {
          resolve({ stdout, stderr, code: e.code });
        } else if (typeof e.code === "string" && e.code.startsWith("ERR_CHILD_PROCESS_STDIO_MAXBUFFER")) {
          resolve({ stdout, stderr: stderr || e.message, code: 1 });
        } else {
          // Spawn failure (e.g. ENOENT) or signal kill.
          resolve({ stdout, stderr: stderr ? `${stderr}\n${e.message}` : e.message, code: SPAWN_FAILURE_CODE });
        }
      },
    );
  });
