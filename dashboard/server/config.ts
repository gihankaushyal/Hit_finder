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

/**
 * An empty or blank setting counts as unset (and a padded one is trimmed): `DASH_X=$UNSET_VAR cmd` passes "",
 * which must not resolve to the current directory. Every setting except the port is read through this.
 */
const given = (v: string | undefined): string | undefined => {
  const t = v?.trim();
  return t ? t : undefined;
};

export function loadConfig(raw: NodeJS.ProcessEnv): Config {
  const get = (key: string) => given(raw[key]);
  let port = DEFAULT_PORT;
  // The port stays strict: a blank value is an error, because silently using the default could bind a port
  // another instance already holds, or send the user to the wrong dashboard.
  if (raw.DASH_PORT !== undefined) {
    const p = raw.DASH_PORT.trim();
    if (!/^\d+$/.test(p) || Number(p) < 1 || Number(p) > 65535) {
      throw new Error(`DASH_PORT must be a port number from 1 to 65535, got "${raw.DASH_PORT}"`);
    }
    port = Number(p);
  }
  const repoRoot = path.resolve(get("DASH_REPO_ROOT") ?? path.dirname(DASHBOARD_DIR));
  return {
    repoRoot,
    host: "127.0.0.1",
    port,
    stateDir: path.resolve(get("DASH_STATE_DIR") ?? path.join(DASHBOARD_DIR, ".state")),
    ghBin: get("DASH_GH_BIN") ?? "gh",
    pythonBin: get("DASH_PYTHON") ?? "python",
    kanbanPath: path.resolve(get("DASH_KANBAN") ?? path.join(repoRoot, "phase-05-kanban.md")),
    checkpointsDir: path.resolve(get("DASH_CHECKPOINTS") ?? path.join(repoRoot, "checkpoints")),
    webDist: path.join(DASHBOARD_DIR, "dist"),
  };
}
