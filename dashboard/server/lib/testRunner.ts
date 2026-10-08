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
const SPAWN_FAILURE_CODE = 127;
const JUNIT_FILE = "junit.xml";
const LAST_STATE_FILE = "tests-last.json";
const STATUSES = new Set(["idle", "running", "done", "error"]);

export interface SpawnHandle {
  /** Called for each stdout and stderr line. */
  onLine(fn: (line: string) => void): void;
  /** Exit code; 127 if the spawn fails. Never rejects. */
  done: Promise<number>;
  /** Terminate the child (SIGTERM, then SIGKILL after a grace period). */
  kill(): void;
}
export type SpawnFn = (cmd: string, args: string[], opts: { cwd: string }) => SpawnHandle;

export interface TestRunner {
  state(): TestRunState;
  /** Returns false if a run is already active. */
  start(): boolean;
  /** Kill the active run's child, if any. Call on server shutdown. */
  stop(): void;
}

const KILL_GRACE_MS = 2000;
const liveChildren = new Set<() => void>();
// Last-resort cleanup: if the server process exits, take any running pytest with it.
process.on("exit", () => {
  for (const kill of liveChildren) kill();
});

/** Long-running spawn with line streaming; deliberately has no timeout (full runs take minutes). */
export const nodeSpawn: SpawnFn = (cmd, args, opts) => {
  const listeners: ((line: string) => void)[] = [];
  const child = nodeSpawnProcess(cmd, args, { cwd: opts.cwd, stdio: ["ignore", "pipe", "pipe"] });
  const forward = (line: string) => {
    for (const fn of listeners) fn(line);
  };
  for (const stream of [child.stdout, child.stderr]) {
    if (stream) readline.createInterface({ input: stream }).on("line", forward);
  }
  const kill = () => {
    if (child.exitCode !== null || child.signalCode !== null) return;
    child.kill("SIGTERM");
    setTimeout(() => child.kill("SIGKILL"), KILL_GRACE_MS).unref();
  };
  liveChildren.add(kill);
  const done = new Promise<number>((resolve) => {
    child.on("error", () => {
      liveChildren.delete(kill);
      resolve(SPAWN_FAILURE_CODE);
    });
    child.on("close", (code, signal) => {
      liveChildren.delete(kill);
      resolve(code ?? (signal ? 128 : SPAWN_FAILURE_CODE));
    });
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
  let kill: (() => void) | null = null;

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
    try {
      fs.mkdirSync(config.stateDir, { recursive: true });
      fs.rmSync(junitPath, { force: true });
      const handle = d.spawn(config.pythonBin, ["-m", "pytest", "tests/", "-q", `--junitxml=${junitPath}`], { cwd: config.repoRoot });
      kill = handle.kill;
      handle.onLine((line) => {
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
      void run();
      return true;
    },
    stop() {
      kill?.();
    },
  };
}
