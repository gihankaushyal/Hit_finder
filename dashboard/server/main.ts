import type { Server } from "node:http";
import { serve } from "@hono/node-server";
import { loadOrCreateToken } from "./auth";
import { loadConfig, type Config } from "./config";
import { EventBus } from "./events";
import { createGh } from "./lib/gh";
import { createKanbanService } from "./lib/kanbanService";
import { fileSyncDeps, isImportCompleted, removeStaleTempFiles } from "./lib/kanbanSync";
import { startKanbanTriggers } from "./lib/kanbanTriggers";
import { createTestRunner, nodeSpawn } from "./lib/testRunner";
import { execRunner } from "./runner";
import { buildServer } from "./wiring";

const EXIT_FAILURE = 1;
/** Open SSE streams never end on their own; force the exit after this long. */
const SHUTDOWN_GRACE_MS = 3000;

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
const app = buildServer({ config, token, bus, gh, tests, kanban, runner: execRunner });

if (!isImportCompleted(config)) {
  console.log("Kanban sync is off until you run the first import: npm run kanban:sync -- --dry-run, then -- --yes");
}
const triggers = startKanbanTriggers(kanban, config.kanbanPath);

const server = serve({ fetch: app.fetch, hostname: config.host, port: config.port }, () => {
  console.log(`http://${config.host}:${config.port}/?token=${token}`);
}) as Server;

server.on("error", (err: NodeJS.ErrnoException) => {
  void triggers.close();
  tests.stop();
  fail(err.code === "EADDRINUSE" ? `dashboard: port ${config.port} is already in use` : `dashboard: could not start the server (${err.code ?? "error"})`);
});

let shuttingDown = false;
function shutdown(): void {
  if (shuttingDown) return;
  shuttingDown = true;
  tests.stop();
  const force = setTimeout(() => process.exit(0), SHUTDOWN_GRACE_MS);
  force.unref();
  void triggers.close().finally(() => {
    server.close(() => process.exit(0));
    server.closeAllConnections?.();
  });
}
process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
