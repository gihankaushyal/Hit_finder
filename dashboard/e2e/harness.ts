import { spawn, spawnSync, type ChildProcess } from "node:child_process";
import fs from "node:fs";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const E2E_DIR = path.dirname(fileURLToPath(import.meta.url));
export const DASHBOARD_DIR = path.resolve(E2E_DIR, "..");
export const FAKE_GH = path.join(E2E_DIR, "fake-gh.mjs");
export const FAKE_PYTEST = path.join(E2E_DIR, "fake-pytest.mjs");
const KANBAN_FIXTURE = path.join(E2E_DIR, "fixtures", "kanban-e2e.md");
const TSX_BIN = path.join(DASHBOARD_DIR, "node_modules", ".bin", "tsx");
const TOKEN_BYTES = 32;
const READY_TIMEOUT_MS = 30_000;
const STOP_GRACE_MS = 5_000;
const POLL_MS = 100;

/** Env vars whose value is a path that must lie inside the run's temp directory. */
const TEMP_PATH_VARS = ["DASH_KANBAN", "DASH_STATE_DIR", "DASH_REPO_ROOT", "FAKE_GH_STATE", "FAKE_GH_CALLS", "FAKE_PYTEST_CALLS"] as const;

const realPath = (p: string): string => {
  try {
    return fs.realpathSync(p);
  } catch {
    return path.resolve(p);
  }
};
const isInside = (root: string, p: string): boolean => {
  const rel = path.relative(realPath(root), path.resolve(p));
  return rel !== "" && !rel.startsWith("..") && !path.isAbsolute(rel);
};

/**
 * Fails fast unless every path the server or CLI will touch lies inside `tmpRoot` and the external
 * programs are the fakes. An e2e run must never reach the real kanban file, state directory or gh.
 */
export function assertSafeEnv(env: Record<string, string | undefined>, tmpRoot: string): void {
  const realKanban = path.resolve(DASHBOARD_DIR, "..", "phase-05-kanban.md");
  const realState = path.join(DASHBOARD_DIR, ".state");
  if (isInside(DASHBOARD_DIR, tmpRoot) || realPath(tmpRoot) === realPath(DASHBOARD_DIR)) {
    throw new Error(`e2e: the temp directory ${tmpRoot} must not be inside the dashboard directory`);
  }
  for (const key of TEMP_PATH_VARS) {
    const v = env[key];
    if (key === "FAKE_PYTEST_CALLS" && v === undefined) continue;
    if (!v) throw new Error(`e2e: ${key} is not set; refusing to start`);
    if (!isInside(tmpRoot, v)) throw new Error(`e2e: ${key}=${v} is not inside the temp directory ${tmpRoot}; refusing to start`);
  }
  if (path.resolve(env.DASH_KANBAN!) === realKanban || path.resolve(env.DASH_STATE_DIR!) === realState) {
    throw new Error("e2e: the configuration points at the real kanban file or state directory; refusing to start");
  }
  if (!env.DASH_GH_BIN || realPath(env.DASH_GH_BIN) !== realPath(FAKE_GH)) {
    throw new Error(`e2e: DASH_GH_BIN must be ${FAKE_GH}, got ${env.DASH_GH_BIN ?? "(unset)"}; refusing to start`);
  }
  if (!env.DASH_PYTHON || realPath(env.DASH_PYTHON) !== realPath(FAKE_PYTEST)) {
    throw new Error(`e2e: DASH_PYTHON must be ${FAKE_PYTEST}, got ${env.DASH_PYTHON ?? "(unset)"}; refusing to start`);
  }
}

export interface GhCall {
  args: string[];
  at: string;
}
export interface CliResult {
  code: number | null;
  stdout: string;
  stderr: string;
}
export interface Dashboard {
  baseUrl: string;
  /** The URL the server prints: sets the cookie, then redirects to the app. */
  tokenUrl: string;
  token: string;
  dir: string;
  kanbanPath: string;
  stateDir: string;
  /** Every fake gh invocation so far, in order. */
  ghCalls(): GhCall[];
  /** The fake gh issue store. */
  ghIssues(): { number: number; title: string; state: string; labels: string[] }[];
  /** Number of times the fake pytest was started. */
  pytestRuns(): number;
  kanbanText(): string;
  /** Make fake gh fail for commands starting with any of these ("pr list", "issue list", ...). */
  failGh(prefixes: string[]): void;
  clearGhFailures(): void;
  /** Runs the kanban sync CLI against this dashboard's temp directory, with the fake gh. */
  runCli(args: string[]): Promise<CliResult>;
  stop(): Promise<void>;
}
export interface StartOptions {
  seed?: "default" | "empty";
  /** Kanban markdown; defaults to the e2e fixture. */
  kanban?: string;
  pytestDelayMs?: number;
}

async function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.once("error", reject);
    srv.listen(0, "127.0.0.1", () => {
      const { port } = srv.address() as net.AddressInfo;
      srv.close(() => resolve(port));
    });
  });
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function waitReady(baseUrl: string, token: string, child: ChildProcess, log: () => string): Promise<void> {
  const deadline = Date.now() + READY_TIMEOUT_MS;
  while (Date.now() < deadline) {
    if (child.exitCode !== null) throw new Error(`e2e: the server exited with code ${child.exitCode} before it was ready:\n${log()}`);
    try {
      const r = await fetch(`${baseUrl}/api/health`, { headers: { Authorization: `Bearer ${token}` } });
      if (r.ok) return;
    } catch {
      // not listening yet
    }
    await sleep(POLL_MS);
  }
  throw new Error(`e2e: the server did not become ready in ${READY_TIMEOUT_MS} ms:\n${log()}`);
}

export async function startDashboard(opts: StartOptions = {}): Promise<Dashboard> {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "dash-e2e-"));
  const repoRoot = path.join(dir, "repo");
  const stateDir = path.join(dir, "state");
  const kanbanPath = path.join(repoRoot, "phase-05-kanban.md");
  const ghState = path.join(dir, "fake-gh.json");
  const ghCallsFile = path.join(dir, "fake-gh-calls.jsonl");
  const pytestCallsFile = path.join(dir, "fake-pytest-calls.txt");
  fs.mkdirSync(repoRoot, { recursive: true });
  fs.mkdirSync(path.join(dir, "home"), { recursive: true });
  fs.writeFileSync(kanbanPath, opts.kanban ?? fs.readFileSync(KANBAN_FIXTURE, "utf8"));
  fs.writeFileSync(ghCallsFile, "");
  fs.writeFileSync(pytestCallsFile, "");
  // A throwaway git repo so the top bar has a branch and commit to show.
  const git = (args: string[]) => spawnSync("git", ["-c", "user.name=e2e", "-c", "user.email=e2e@example.invalid", ...args], { cwd: repoRoot, encoding: "utf8" });
  git(["init", "-q", "-b", "e2e-branch"]);
  git(["commit", "-q", "--allow-empty", "-m", "e2e"]);

  const token = Buffer.from(Array.from({ length: TOKEN_BYTES }, (_, i) => (i * 37 + 11) & 0xff)).toString("hex");
  fs.mkdirSync(stateDir, { recursive: true, mode: 0o700 });
  fs.writeFileSync(path.join(stateDir, "token"), token, { mode: 0o600 });

  const port = await freePort();
  const env: Record<string, string> = {
    PATH: process.env.PATH ?? "",
    HOME: path.join(dir, "home"),
    TMPDIR: dir,
    NODE_ENV: "production",
    DASH_PORT: String(port),
    DASH_STATE_DIR: stateDir,
    DASH_REPO_ROOT: repoRoot,
    DASH_KANBAN: kanbanPath,
    DASH_GH_BIN: FAKE_GH,
    DASH_PYTHON: FAKE_PYTEST,
    FAKE_GH_STATE: ghState,
    FAKE_GH_CALLS: ghCallsFile,
    FAKE_GH_SEED: opts.seed ?? "default",
    FAKE_PYTEST_CALLS: pytestCallsFile,
    FAKE_PYTEST_DELAY_MS: String(opts.pytestDelayMs ?? 700),
  };
  try {
    assertSafeEnv(env, dir);
  } catch (err) {
    fs.rmSync(dir, { recursive: true, force: true });
    throw err;
  }

  let output = "";
  const child = spawn(TSX_BIN, ["server/main.ts"], { cwd: DASHBOARD_DIR, env, stdio: ["ignore", "pipe", "pipe"] });
  child.stdout?.on("data", (d: Buffer) => (output += d.toString()));
  child.stderr?.on("data", (d: Buffer) => (output += d.toString()));
  const baseUrl = `http://127.0.0.1:${port}`;

  const stop = async (): Promise<void> => {
    if (child.exitCode === null && child.signalCode === null) {
      const exited = new Promise<void>((r) => child.once("exit", () => r()));
      child.kill("SIGTERM");
      const killer = setTimeout(() => child.kill("SIGKILL"), STOP_GRACE_MS);
      await exited;
      clearTimeout(killer);
    }
    fs.rmSync(dir, { recursive: true, force: true });
  };
  try {
    await waitReady(baseUrl, token, child, () => output);
  } catch (err) {
    await stop();
    throw err;
  }

  const readState = (): { issues: { number: number; title: string; state: string; labels: string[] }[] } => {
    try {
      return JSON.parse(fs.readFileSync(ghState, "utf8"));
    } catch {
      return { issues: [] };
    }
  };
  return {
    baseUrl,
    tokenUrl: `${baseUrl}/?token=${token}`,
    token,
    dir,
    kanbanPath,
    stateDir,
    ghCalls: () =>
      fs.readFileSync(ghCallsFile, "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l) as GhCall),
    ghIssues: () => readState().issues,
    pytestRuns: () => fs.readFileSync(pytestCallsFile, "utf8").split("\n").filter(Boolean).length,
    kanbanText: () => fs.readFileSync(kanbanPath, "utf8"),
    failGh: (prefixes) => fs.writeFileSync(`${ghState}.fail`, JSON.stringify(prefixes)),
    clearGhFailures: () => fs.rmSync(`${ghState}.fail`, { force: true }),
    runCli(args) {
      assertSafeEnv(env, dir);
      return new Promise((resolve, reject) => {
        const cli = spawn(TSX_BIN, ["server/cli/kanbanSync.ts", ...args], { cwd: DASHBOARD_DIR, env, stdio: ["ignore", "pipe", "pipe"] });
        let stdout = "";
        let stderr = "";
        cli.stdout?.on("data", (d: Buffer) => (stdout += d.toString()));
        cli.stderr?.on("data", (d: Buffer) => (stderr += d.toString()));
        cli.once("error", reject);
        cli.once("exit", (code) => resolve({ code, stdout, stderr }));
      });
    },
    stop,
  };
}
