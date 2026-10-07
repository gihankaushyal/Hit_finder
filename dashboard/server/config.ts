import path from "node:path";
import { fileURLToPath } from "node:url";

export interface Config {
  repoRoot: string;   // DASH_REPO_ROOT, default: parent of dashboard/
  host: "127.0.0.1";
  port: number;       // DASH_PORT, default 4317
  stateDir: string;   // DASH_STATE_DIR, default dashboard/.state
  ghBin: string;      // DASH_GH_BIN, default "gh"
  pythonBin: string;  // DASH_PYTHON, default "python"
  kanbanPath: string; // DASH_KANBAN, default <repoRoot>/phase-05-kanban.md
  webDist: string;    // dashboard/dist
}

const DEFAULT_PORT = 4317;
// server/config.ts -> dashboard/ -> repo root
const DASHBOARD_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

export function loadConfig(env: NodeJS.ProcessEnv): Config {
  let port = DEFAULT_PORT;
  if (env.DASH_PORT !== undefined) {
    if (!/^\d+$/.test(env.DASH_PORT) || Number(env.DASH_PORT) > 65535) {
      throw new Error(`DASH_PORT must be a valid port number, got "${env.DASH_PORT}"`);
    }
    port = Number(env.DASH_PORT);
  }
  const repoRoot = env.DASH_REPO_ROOT ?? path.dirname(DASHBOARD_DIR);
  return {
    repoRoot,
    host: "127.0.0.1",
    port,
    stateDir: env.DASH_STATE_DIR ?? path.join(DASHBOARD_DIR, ".state"),
    ghBin: env.DASH_GH_BIN ?? "gh",
    pythonBin: env.DASH_PYTHON ?? "python",
    kanbanPath: env.DASH_KANBAN ?? path.join(repoRoot, "phase-05-kanban.md"),
    webDist: path.join(DASHBOARD_DIR, "dist"),
  };
}
