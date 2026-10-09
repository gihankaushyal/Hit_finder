import { spawn as nodeSpawnProcess } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import readline from "node:readline";
import type { Config } from "../config";
import type { EventBus } from "../events";
import type { Runner } from "../runner";
import type { TestRunState } from "../../shared/types";
import { parseJunit } from "./junit";

const TAIL_MAX_LINES = 200;
const LINE_MAX_CHARS = 2000;
const SPAWN_FAILURE_CODE = 127;
const JUNIT_FILE = "junit.xml";
const LAST_STATE_FILE = "tests-last.json";
const STATUSES = new Set(["idle", "running", "done", "error"]);

export interface SpawnHandle {
  /** Called for each stdout and stderr line. */
  onLine(fn: (line: string) => void): void;
  /** Exit code; 127 if the spawn fails. Never rejects. */
  done: Promise<number>;
  /** Terminate the child's process group (SIGTERM, then SIGKILL after a grace period); resolves once the group is gone. */
  kill(): Promise<void>;
}
export type SpawnFn = (cmd: string, args: string[], opts: { cwd: string }) => SpawnHandle;

export interface TestRunner {
  state(): TestRunState;
  /** Returns false if a run is already active. */
  start(): boolean;
  /**
   * Cancel the active run. A run that has not spawned yet never will; a running child is
   * killed. Resolves once the child process group is gone. Call on server shutdown.
   */
  stop(): Promise<void>;
}

const KILL_GRACE_MS = 2000;
const KILL_POLL_MS = 25;
/** After the child exits, wait this long for output to drain before giving up on leftover pipe holders. */
const DRAIN_MS = 1000;
const liveChildren = new Set<() => Promise<void>>();
// Last-resort cleanup: if the server process exits, take any running pytest with it.
process.on("exit", () => {
  for (const kill of liveChildren) void kill();
});

/** Long-running spawn with line streaming; deliberately has no timeout (full runs take minutes). */
export const nodeSpawn: SpawnFn = (cmd, args, opts) => {
  const listeners: ((line: string) => void)[] = [];
  // detached: the child leads its own process group, so kill() reaches grandchildren too.
  const child = nodeSpawnProcess(cmd, args, { cwd: opts.cwd, stdio: ["ignore", "pipe", "pipe"], detached: true });
  const forward = (line: string) => {
    for (const fn of listeners) fn(line);
  };
  for (const stream of [child.stdout, child.stderr]) {
    if (stream) readline.createInterface({ input: stream }).on("line", forward);
  }
  const signalGroup = (signal: NodeJS.Signals) => {
    if (child.pid === undefined) return;
    try {
      process.kill(-child.pid, signal);
    } catch {
      // group already gone
    }
  };
  const groupAlive = (): boolean => {
    if (child.pid === undefined) return false;
    try {
      process.kill(-child.pid, 0);
      return true;
    } catch (err) {
      return (err as NodeJS.ErrnoException).code === "EPERM";
    }
  };
  let killing: Promise<void> | null = null;
  const kill = (): Promise<void> => {
    killing ??= new Promise<void>((resolve) => {
      if (!groupAlive()) return resolve();
      signalGroup("SIGTERM");
      // The leader may exit on SIGTERM while a grandchild ignores it, so escalate on the group.
      const started = Date.now();
      let escalated = false;
      const poll = setInterval(() => {
        const elapsed = Date.now() - started;
        if (!groupAlive() || elapsed >= 2 * KILL_GRACE_MS + KILL_POLL_MS) {
          clearInterval(poll);
          resolve();
        } else if (!escalated && elapsed >= KILL_GRACE_MS) {
          escalated = true;
          signalGroup("SIGKILL");
        }
      }, KILL_POLL_MS);
    });
    return killing;
  };
  liveChildren.add(kill);
  const done = new Promise<number>((resolve) => {
    let settled = false;
    const settle = (code: number) => {
      if (settled) return;
      settled = true;
      liveChildren.delete(kill);
      resolve(code);
    };
    child.on("error", () => settle(SPAWN_FAILURE_CODE));
    child.on("exit", (code, signal) => {
      const result = code ?? (signal ? 128 : SPAWN_FAILURE_CODE);
      // `close` waits for every holder of the pipes; a leftover background process must not hang the run.
      setTimeout(() => {
        child.stdout?.destroy();
        child.stderr?.destroy();
        settle(result);
      }, DRAIN_MS).unref();
    });
    child.on("close", (code, signal) => settle(code ?? (signal ? 128 : SPAWN_FAILURE_CODE)));
  });
  return { onLine: (fn) => listeners.push(fn), done, kill };
};

const idleState = (): TestRunState => ({
  status: "idle", startedAt: null, finishedAt: null, commit: null, summary: null, tail: [], error: null,
});

function loadPersisted(file: string): TestRunState {
  try {
    const raw = JSON.parse(fs.readFileSync(file, "utf8")) as Partial<TestRunState>;
    if (raw && typeof raw === "object" && typeof raw.status === "string" && STATUSES.has(raw.status) && raw.status !== "running") {
      return { ...idleState(), ...raw } as TestRunState;
    }
  } catch {
    // missing or corrupt: start idle
  }
  return idleState();
}

export function createTestRunner(d: { config: Config; spawn: SpawnFn; runner: Runner; bus: EventBus }): TestRunner {
  const { config, bus } = d;
  const junitPath = path.join(config.stateDir, JUNIT_FILE);
  const lastPath = path.join(config.stateDir, LAST_STATE_FILE);
  let current: TestRunState = loadPersisted(lastPath);
  let running = false;
  let kill: (() => Promise<void>) | null = null;
  let cancelled = false;
  let runDone: Promise<void> = Promise.resolve();

  const invalidate = () => bus.emit({ type: "invalidate", resource: "tests" });

  function persist(): void {
    try {
      fs.mkdirSync(config.stateDir, { recursive: true });
      const tmp = `${lastPath}.tmp`;
      fs.writeFileSync(tmp, JSON.stringify(current));
      fs.renameSync(tmp, lastPath);
    } catch (err) {
      console.error("could not persist test state", err);
    }
  }

  function finish(patch: Partial<TestRunState>): void {
    current = { ...current, ...patch, finishedAt: new Date().toISOString() };
    running = false;
    kill = null;
    cancelled = false;
    persist();
    invalidate();
  }

  async function run(): Promise<void> {
    let commit: string | null = null;
    try {
      const r = await d.runner("git", ["rev-parse", "--short", "HEAD"], { cwd: config.repoRoot });
      if (r.code === 0) commit = r.stdout.trim() || null;
    } catch {
      commit = null;
    }
    current = { ...current, commit };
    if (cancelled) {
      finish({ status: "error", summary: null, error: "the test run was cancelled before it started" });
      return;
    }
    try {
      fs.mkdirSync(config.stateDir, { recursive: true });
      fs.rmSync(junitPath, { force: true });
      const handle = d.spawn(config.pythonBin, ["-m", "pytest", "tests/", "-q", `--junitxml=${junitPath}`], { cwd: config.repoRoot });
      kill = handle.kill;
      if (cancelled) void handle.kill();
      handle.onLine((raw) => {
        const line = raw.length > LINE_MAX_CHARS ? raw.slice(0, LINE_MAX_CHARS) : raw;
        current = { ...current, tail: [...current.tail, line].slice(-TAIL_MAX_LINES) };
        bus.emit({ type: "tests-line", line });
      });
      const code = await handle.done;
      if (!fs.existsSync(junitPath)) {
        finish({ status: "error", summary: null, error: `pytest exited with code ${code} and wrote no report` });
        return;
      }
      try {
        finish({ status: "done", summary: parseJunit(fs.readFileSync(junitPath, "utf8")), error: null });
      } catch (err) {
        console.error("could not parse junit report", err);
        finish({ status: "error", summary: null, error: "could not read the pytest report" });
      }
    } catch (err) {
      console.error("test run failed", err);
      finish({ status: "error", summary: null, error: "could not start pytest" });
    }
  }

  return {
    state: () => current,
    start() {
      if (running) return false;
      running = true;
      current = { ...idleState(), status: "running", startedAt: new Date().toISOString() };
      invalidate();
      cancelled = false;
      runDone = run();
      return true;
    },
    async stop() {
      if (!running) return;
      cancelled = true;
      if (kill) await kill();
      else await runDone; // not spawned yet: run() sees the flag and never spawns
    },
  };
}
