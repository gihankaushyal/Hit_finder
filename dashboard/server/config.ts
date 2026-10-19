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
  checkpointsDir: string; // DASH_CHECKPOINTS, default <repoRoot>/checkpoints
  webDist: string;    // dashboard/dist
}

const DEFAULT_PORT = 4317;
// server/config.ts -> dashboard/ -> repo root
const DASHBOARD_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

/** An empty or blank setting counts as unset: `DASH_X=$UNSET_VAR cmd` passes "", which must not become the cwd. */
const given = (v: string | undefined): string | undefined => (v === undefined || v.trim() === "" ? undefined : v);

export function loadConfig(raw: NodeJS.ProcessEnv): Config {
  const env = {
    DASH_PORT: given(raw.DASH_PORT),
    DASH_REPO_ROOT: given(raw.DASH_REPO_ROOT),
    DASH_STATE_DIR: given(raw.DASH_STATE_DIR),
    DASH_GH_BIN: given(raw.DASH_GH_BIN),
    DASH_PYTHON: given(raw.DASH_PYTHON),
    DASH_KANBAN: given(raw.DASH_KANBAN),
    DASH_CHECKPOINTS: given(raw.DASH_CHECKPOINTS),
  };
  let port = DEFAULT_PORT;
  if (env.DASH_PORT !== undefined) {
    if (!/^\d+$/.test(env.DASH_PORT) || Number(env.DASH_PORT) < 1 || Number(env.DASH_PORT) > 65535) {
      throw new Error(`DASH_PORT must be a port number from 1 to 65535, got "${env.DASH_PORT}"`);
    }
    port = Number(env.DASH_PORT);
  }
  const repoRoot = path.resolve(env.DASH_REPO_ROOT ?? path.dirname(DASHBOARD_DIR));
  return {
    repoRoot,
    host: "127.0.0.1",
    port,
    stateDir: path.resolve(env.DASH_STATE_DIR ?? path.join(DASHBOARD_DIR, ".state")),
    ghBin: env.DASH_GH_BIN ?? "gh",
    pythonBin: env.DASH_PYTHON ?? "python",
    kanbanPath: path.resolve(env.DASH_KANBAN ?? path.join(repoRoot, "phase-05-kanban.md")),
    checkpointsDir: path.resolve(env.DASH_CHECKPOINTS ?? path.join(repoRoot, "checkpoints")),
    webDist: path.join(DASHBOARD_DIR, "dist"),
  };
}
