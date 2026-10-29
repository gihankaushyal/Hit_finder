import type { Server } from "node:http";
import { serve } from "@hono/node-server";
import path from "node:path";
import { createApprovals } from "./agents/approvals";
import { createChief } from "./agents/chief";
import { createChiefTools } from "./agents/chiefTools";
import { chiefScripts, createFakeDriver } from "./agents/fakeDriver";
import { sweepOrphans } from "./agents/orphans";
import { createSdkDriver } from "./agents/sdkDriver";
import { createSupervisor } from "./agents/supervisor";
import { createWorktrees } from "./agents/worktrees";
import { loadOrCreateToken } from "./auth";
import { createTerminal } from "./terminal/terminal";
import { loadConfig, type Config } from "./config";
import { EventBus } from "./events";
import { createGh } from "./lib/gh";
import { createKanbanService } from "./lib/kanbanService";
import { fileSyncDeps, isImportCompleted, removeStaleTempFiles } from "./lib/kanbanSync";
import { startKanbanTriggers } from "./lib/kanbanTriggers";
import { createRunScanner } from "./lib/runScan";
import { createTestRunner, nodeSpawn } from "./lib/testRunner";
import { execRunner } from "./runner";
import { buildServer } from "./wiring";

const EXIT_FAILURE = 1;
/** Open SSE streams never end on their own; force the exit after this long. */
const SHUTDOWN_GRACE_MS = 3000;
/** Upper bound on waiting for pytest's process group to die (SIGTERM, then SIGKILL after 2 s) before exiting. */
const STOP_TESTS_TIMEOUT_MS = 6000;
/** Upper bound on closing the terminal streams at shutdown (the tmux session itself is left running). */
const STOP_TERMINAL_TIMEOUT_MS = 3000;
/** Upper bound on stopping every agent session (interrupt, close, process-group kill) at shutdown. */
const STOP_AGENTS_TIMEOUT_MS = 5000;

function stopTests(): Promise<void> {
  const bound = new Promise<void>((resolve) => setTimeout(resolve, STOP_TESTS_TIMEOUT_MS).unref());
  return Promise.race([tests.stop().catch(() => {}), bound]);
}

function fail(message: string): never {
  console.error(message);
  process.exit(EXIT_FAILURE);
}

let config: Config;
try {
  config = loadConfig(process.env);
} catch (err) {
  fail(`dashboard: ${err instanceof Error ? err.message : "invalid configuration"}`);
}

removeStaleTempFiles(config);
const token = loadOrCreateToken(config.stateDir);
const bus = new EventBus();
const gh = createGh(execRunner, config);
const tests = createTestRunner({ config, spawn: nodeSpawn, runner: execRunner, bus });
const kanban = createKanbanService({
  gh,
  sync: fileSyncDeps(config, gh),
  bus,
  imported: () => isImportCompleted(config),
});
const runs = createRunScanner(config.checkpointsDir);
const approvals = createApprovals({ bus, auditPath: path.join(config.stateDir, "approvals-audit.jsonl") });
const agents = createSupervisor({
  cfg: config,
  bus,
  driver: config.agentDriver === "fake" ? createFakeDriver({ stepMs: 150 }) : createSdkDriver(),
  approvals,
  worktrees: createWorktrees({ repoRoot: config.repoRoot, agentsDir: config.agentsDir, base: config.agentBase }),
  env: process.env,
  sweep: async () => void (await sweepOrphans(config.agentsDir)),
});
const chief = createChief({
  cfg: config,
  bus,
  driver: config.agentDriver === "fake" ? createFakeDriver({ stepMs: 150, scripts: chiefScripts }) : createSdkDriver(),
  approvals,
  tools: createChiefTools({ supervisor: agents, kanban }),
  env: process.env,
  // The fake driver's sessions have no transcript on disk, so they must not be mistaken for expired ones.
  transcriptExists: config.agentDriver === "fake" ? async () => true : undefined,
});
const terminal = createTerminal({ cfg: config, env: process.env });
const app = buildServer({ config, token, bus, gh, tests, kanban, runner: execRunner, runs, agents, approvals, chief, terminal });

if (!isImportCompleted(config)) {
  console.log("Kanban sync is off until you run the first import: npm run kanban:sync -- --dry-run, then -- --yes");
}
const triggers = startKanbanTriggers(kanban, config.kanbanPath);
const startFailed = (what: string) => (err: unknown) => console.error(`${what} failed to start:`, err instanceof Error ? err.message : err);
const agentsReady = Promise.all([agents.init().catch(startFailed("agent supervisor")), chief.init().catch(startFailed("chief of staff"))]);

const server = serve({ fetch: app.fetch, hostname: config.host, port: config.port }, () => {
  console.log(`http://${config.host}:${config.port}/?token=${token}`);
}) as Server;

server.on("error", (err: NodeJS.ErrnoException) => {
  void triggers.close();
  const message = err.code === "EADDRINUSE" ? `dashboard: port ${config.port} is already in use` : `dashboard: could not start the server (${err.code ?? "error"})`;
  void Promise.all([terminal.closeAll(STOP_TERMINAL_TIMEOUT_MS), stopTests(), agentsReady.then(() => Promise.all([chief.stop(), agents.stopAll(STOP_AGENTS_TIMEOUT_MS)]))]).finally(() => fail(message));
});

let shuttingDown = false;
function shutdown(): void {
  if (shuttingDown) return;
  shuttingDown = true;
  const force = setTimeout(() => process.exit(0), STOP_TESTS_TIMEOUT_MS + STOP_AGENTS_TIMEOUT_MS + STOP_TERMINAL_TIMEOUT_MS + SHUTDOWN_GRACE_MS);
  force.unref();
  // Exit only after pytest's process group is gone, or the bound above elapses.
  const testsStopped = Promise.all([terminal.closeAll(STOP_TERMINAL_TIMEOUT_MS), stopTests(), agentsReady.then(() => Promise.all([chief.stop(), agents.stopAll(STOP_AGENTS_TIMEOUT_MS)]))]);
  chief.dispose();
  void triggers.close().finally(() => {
    server.close(() => void testsStopped.finally(() => process.exit(0)));
    server.closeAllConnections?.();
  });
}
process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
