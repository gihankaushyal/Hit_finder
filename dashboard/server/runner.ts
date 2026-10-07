import { execFile } from "node:child_process";

export interface RunResult { stdout: string; stderr: string; code: number }
export type Runner = (cmd: string, args: string[], opts?: { cwd?: string }) => Promise<RunResult>;

const MAX_BUFFER_BYTES = 32 * 1024 * 1024;
const SPAWN_FAILURE_CODE = 127;

export const execRunner: Runner = (cmd, args, opts) =>
  new Promise((resolve) => {
    execFile(
      cmd,
      args,
      { cwd: opts?.cwd, maxBuffer: MAX_BUFFER_BYTES, encoding: "utf8" },
      (error, stdout, stderr) => {
        if (!error) {
          resolve({ stdout, stderr, code: 0 });
          return;
        }
        const e = error as NodeJS.ErrnoException & { code?: number | string };
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
