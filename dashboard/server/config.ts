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
  agentsDir: string;      // DASH_AGENTS_DIR, default <repoRoot>-agents (worker worktrees live here)
  claudeBin: string;      // DASH_CLAUDE_BIN, default "claude" on PATH
  agentMax: number;       // DASH_AGENT_MAX, default 3 workers at once (1 to 8)
  agentMaxTurns: number;  // DASH_AGENT_MAX_TURNS, default 60 per worker (1 to 500)
  workerModel: string;    // DASH_WORKER_MODEL, default sonnet
  chiefModel: string;     // DASH_CHIEF_MODEL, default opus
  agentDriver: "sdk" | "fake"; // DASH_AGENT_DRIVER, default sdk; fake is for tests only
  terminalEnabled: boolean;      // DASH_TERMINAL, on (default) or off
  terminalSession: string;       // DASH_TERMINAL_SESSION, the tmux session the terminal attaches to
  terminalMax: number;           // DASH_TERMINAL_MAX, live terminal streams at once (1 to 4)
  terminalPython: string;        // DASH_TERMINAL_PYTHON, interpreter for the PTY helper
  terminalCommand: string[] | null; // DASH_TERMINAL_COMMAND, tests only: replaces the tmux command (words split on spaces)
  terminalTmuxSocket: string | null; // DASH_TERMINAL_TMUX_SOCKET, tests only: a private tmux socket name
  agentBase: string;      // DASH_AGENT_BASE, the ref worker branches start from, default origin/phase-05
  webDist: string;    // dashboard/dist
}

const DEFAULT_PORT = 4317;
const DEFAULT_AGENT_MAX = 3;
const AGENT_MAX_LIMIT = 8;
const DEFAULT_AGENT_MAX_TURNS = 60;
const AGENT_MAX_TURNS_LIMIT = 500;

/** A whole number from 1 to `limit`, or the default for an unset setting; anything else is an error naming the setting. */
function boundedInt(name: string, value: string | undefined, fallback: number, limit: number): number {
  if (value === undefined) return fallback;
  const n = Number(value);
  if (!/^\d+$/.test(value) || n < 1 || n > limit) throw new Error(`${name} must be a whole number from 1 to ${limit}, got "${value}"`);
  return n;
}
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

const SAFE_NAME = /^[A-Za-z0-9_-]{1,40}$/;
const DEFAULT_TERMINAL_MAX = 2;
const TERMINAL_MAX_LIMIT = 4;

function terminalSwitch(value: string | undefined): boolean {
  if (value === undefined || value === "on") return true;
  if (value === "off") return false;
  throw new Error(`DASH_TERMINAL must be "on" or "off", got "${value}"`);
}

function safeName(name: string, value: string | undefined, fallback: string | null): string | null {
  if (value === undefined) return fallback;
  if (!SAFE_NAME.test(value)) throw new Error(`${name} may contain only letters, digits, hyphens and underscores (1 to 40 characters), got "${value}"`);
  return value;
}

function agentDriver(value: string | undefined): "sdk" | "fake" {
  if (value === undefined) return "sdk";
  if (value === "sdk" || value === "fake") return value;
  throw new Error(`DASH_AGENT_DRIVER must be "sdk" or "fake", got "${value}"`);
}

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
    agentsDir: path.resolve(get("DASH_AGENTS_DIR") ?? `${repoRoot}-agents`),
    claudeBin: get("DASH_CLAUDE_BIN") ?? "claude",
    agentMax: boundedInt("DASH_AGENT_MAX", get("DASH_AGENT_MAX"), DEFAULT_AGENT_MAX, AGENT_MAX_LIMIT),
    agentMaxTurns: boundedInt("DASH_AGENT_MAX_TURNS", get("DASH_AGENT_MAX_TURNS"), DEFAULT_AGENT_MAX_TURNS, AGENT_MAX_TURNS_LIMIT),
    workerModel: get("DASH_WORKER_MODEL") ?? "sonnet",
    chiefModel: get("DASH_CHIEF_MODEL") ?? "opus",
    agentDriver: agentDriver(get("DASH_AGENT_DRIVER")),
    agentBase: get("DASH_AGENT_BASE") ?? "origin/phase-05",
    terminalEnabled: terminalSwitch(get("DASH_TERMINAL")),
    terminalSession: safeName("DASH_TERMINAL_SESSION", get("DASH_TERMINAL_SESSION"), "hitfinder-dash") as string,
    terminalMax: boundedInt("DASH_TERMINAL_MAX", get("DASH_TERMINAL_MAX"), DEFAULT_TERMINAL_MAX, TERMINAL_MAX_LIMIT),
    terminalPython: get("DASH_TERMINAL_PYTHON") ?? "python3",
    terminalCommand: get("DASH_TERMINAL_COMMAND")?.split(/\s+/) ?? null,
    terminalTmuxSocket: safeName("DASH_TERMINAL_TMUX_SOCKET", get("DASH_TERMINAL_TMUX_SOCKET"), null),
    webDist: path.join(DASHBOARD_DIR, "dist"),
  };
}
