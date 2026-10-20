import type { Config } from "../config";
import { mcpToolName } from "./driver";

/** The only free dashboard tool; every other one changes something and raises an approval card. */
export const CHIEF_FREE_TOOLS = ["list_workers"] as const;
export const CHIEF_ASK_TOOLS = ["spawn_worker", "message_worker", "stop_worker", "create_task", "update_task"] as const;

/**
 * What an agent session may do. Pure data: the SDK driver turns it into SDK options and the
 * confinement (server/agents/bwrap.ts) is applied separately.
 *
 * Gated actions are ASK rules, never allow rules: the SDK does not call `canUseTool` for a tool
 * an allow rule already covers. Anything no rule matches also falls through to `canUseTool`,
 * which shows an approval card, so the default is to ask.
 */
export interface PermissionRules {
  allow: string[];
  ask: string[];
  deny: string[];
}

export interface AgentSessionOptions {
  cwd: string;
  model: string;
  maxTurns?: number;
  env: Record<string, string>;
  settingSources: [];
  strictMcpConfig: true;
  extraArgs: { "safe-mode": null };
  permissionMode: "default";
  settings: { permissions: PermissionRules };
}

const ENV_KEYS = ["PATH", "HOME", "USER", "LANG", "TMPDIR", "SHELL"] as const;

/** The child environment: nothing but the basics. No tokens, W&B keys, DASH_* settings or agent sockets. */
export function minimalEnv(src: Record<string, string | undefined>): Record<string, string> {
  const out: Record<string, string> = {};
  for (const k of ENV_KEYS) {
    const v = src[k];
    if (v !== undefined) out[k] = v;
  }
  return out;
}

/** `Bash(cmd)` and `Bash(cmd *)`: the command alone and with arguments. */
const bash = (...cmds: string[]): string[] => cmds.flatMap((c) => [`Bash(${c})`, `Bash(${c} *)`]);

const SECRET_PATHS = ["//**/.secrets/**", "//home/**/.ssh/**", "//home/**/.claude/.credentials*", "//home/**/.config/gh/**", "//home/**/.config/overleaf-mcp/**"];
const HOME_ONLY_EDIT_DENY = ["//**/.secrets/**", "//home/**/.ssh/**"];

export function permissionRules(): PermissionRules {
  return {
    allow: [
      "Read", "Edit", "Write", "Glob", "Grep",
      ...bash("git status", "git diff", "git log", "git show", "git add", "git commit", "git rev-parse", "git branch --show-current"),
      ...bash("ls"),
      ...bash("npm test", "npm run build", "npm run typecheck", "npm ci"),
      ...bash("python -m pytest", "black"),
      ...bash("gh pr view", "gh pr list", "gh pr diff", "gh pr checks", "gh pr status"),
      ...bash("gh issue view", "gh issue list"),
      ...bash("gh run list", "gh run view"),
    ],
    ask: [
      ...bash("git push"),
      ...bash("gh pr create", "gh pr merge", "gh pr comment", "gh pr review", "gh pr close", "gh pr edit", "gh pr ready", "gh pr reopen"),
      ...bash("gh issue create", "gh issue comment", "gh issue close", "gh issue edit", "gh issue reopen", "gh issue delete"),
      ...bash("gh api", "gh release", "gh repo", "gh workflow", "gh run rerun", "gh run cancel"),
      ...bash("sbatch", "scancel", "srun", "salloc", "ssh", "scp", "rsync", "curl", "wget", "rm"),
      ...bash("git update-ref", "git branch -f", "git branch -D", "git branch -d", "git branch -m", "git reset", "git checkout", "git switch", "git worktree", "git config", "git remote", "git gc", "git prune", "git tag", "git submodule"),
    ],
    deny: [...SECRET_PATHS.map((p) => `Read(${p})`), ...HOME_ONLY_EDIT_DENY.map((p) => `Edit(${p})`)],
  };
}

const COMPOUND = /&&|\|\||;|\||\n/;

function patternToRegExp(pattern: string): RegExp {
  const esc = (s: string) => s.replace(/[.+?^${}()|[\]\\]/g, "\\$&");
  if (pattern.endsWith(" *")) return new RegExp(`^${esc(pattern.slice(0, -2)).replace(/\*/g, ".*")}( .*)?$`);
  return new RegExp(`^${esc(pattern).replace(/\*/g, ".*")}$`);
}

/**
 * Test helper that mirrors how Claude Code matches rules: `Tool(pattern)` with `*` wildcards, a trailing
 * ` *` meaning "optionally followed by arguments", and compound shell commands split into their parts.
 * An allow rule must cover every part; an ask or deny rule triggers when it matches any part.
 */
export function matchesRule(rule: string, tool: string, command: string, mode: "allow" | "any" = "any"): boolean {
  const m = /^([A-Za-z]+)(?:\((.*)\))?$/.exec(rule);
  if (!m || m[1] !== tool) return false;
  if (m[2] === undefined) return true;
  const parts = command.split(COMPOUND).map((p) => p.trim()).filter(Boolean);
  const re = patternToRegExp(m[2]);
  return mode === "allow" ? parts.length > 0 && parts.every((p) => re.test(p)) : parts.some((p) => re.test(p));
}

const base = (cfg: Config, o: { cwd: string; env: Record<string, string | undefined>; model?: string }): Omit<AgentSessionOptions, "model" | "settings" | "maxTurns"> => ({
  cwd: o.cwd,
  env: minimalEnv(o.env),
  settingSources: [],
  strictMcpConfig: true,
  extraArgs: { "safe-mode": null },
  permissionMode: "default",
});

export function workerOptions(cfg: Config, o: { cwd: string; env: Record<string, string | undefined>; model?: string }): AgentSessionOptions {
  return { ...base(cfg, o), model: o.model ?? cfg.workerModel, maxTurns: cfg.agentMaxTurns, settings: { permissions: permissionRules() } };
}

/** Chief of staff: reads the repo, runs no commands and edits nothing. Its in-process MCP tools arrive with M2b. */
export function chiefOptions(cfg: Config, o: { cwd: string; env: Record<string, string | undefined>; model?: string }): AgentSessionOptions {
  const rules = permissionRules();
  return {
    ...base(cfg, o),
    model: o.model ?? cfg.chiefModel,
    settings: { permissions: { allow: ["Read", "Grep", "Glob", ...CHIEF_FREE_TOOLS.map(mcpToolName)], ask: CHIEF_ASK_TOOLS.map(mcpToolName), deny: ["Bash", "Edit", "Write", "NotebookEdit", ...rules.deny.filter((r) => r.startsWith("Read("))] } },
  };
}
