import { describe, it, expect } from "vitest";
import { workerOptions, chiefOptions, minimalEnv, permissionRules, matchesRule } from "../server/agents/policy";
import { loadConfig } from "../server/config";

const cfg = loadConfig({ DASH_REPO_ROOT: "/r/repo", DASH_AGENTS_DIR: "/r/agents" });

/** Gated commands: each must be an ask rule and no allow rule may cover it. */
const GATED = [
  "git push origin HEAD",
  "git push",
  "git add a && git commit -m x && git push",
  "gh pr create --title x",
  "gh pr merge 12",
  "gh pr comment 3 --body hi",
  "gh pr review 3",
  "gh pr close 3",
  "gh pr edit 3 --body x",
  "gh api -X PATCH repos/o/r/pulls/3",
  "gh issue create --title x",
  "gh issue comment 3",
  "gh issue close 3",
  "sbatch scripts/x.sh",
  "scancel 123",
  "srun --pty bash",
  "salloc -N1",
  "ssh host",
  "scp a b:",
  "rsync -a a b",
  "curl https://example.com",
  "wget https://example.com",
  "rm -rf build",
  "git update-ref refs/heads/phase-05 HEAD",
  "git branch -f phase-05 HEAD",
  "git branch -D other",
  "git reset --hard HEAD~1",
  "git checkout phase-05",
  "git switch phase-05",
  "git worktree add ../x",
  "git config user.name x",
  "git remote add o u",
  "git gc",
];

/** Commands a worker may run without asking. */
const FREE = [
  "git status",
  "git status --short",
  "git diff",
  "git log --oneline -5",
  "git show HEAD",
  "git add dashboard/x.ts",
  'git commit -m "feat: x"',
  "git rev-parse --short HEAD",
  "ls -la",
  "npm test",
  "npm run build",
  "npm run typecheck",
  "npm ci",
  "python -m pytest tests/ -q",
  "black src/ tests/",
  "gh pr view 3",
  "gh pr list",
  "gh pr diff 3",
  "gh pr checks 3",
  "gh issue view 3",
  "gh issue list",
  "gh run list",
  "gh run view 5",
];

describe("permission rules", () => {
  const rules = permissionRules();

  it.each(GATED)("asks for %s", (cmd) => {
    expect(rules.ask.some((r) => matchesRule(r, "Bash", cmd))).toBe(true);
    expect(rules.allow.some((r) => matchesRule(r, "Bash", cmd, "allow"))).toBe(false);
  });

  it.each(FREE)("allows %s without asking", (cmd) => {
    expect(rules.allow.some((r) => matchesRule(r, "Bash", cmd, "allow"))).toBe(true);
    expect(rules.ask.some((r) => matchesRule(r, "Bash", cmd))).toBe(false);
  });

  it("never allows a generic reader or shell", () => {
    for (const cmd of ["cat ~/.ssh/id_rsa", "head x", "env", "python -c 'print(1)'", "sh -c 'id'", "bash -c id", "node -e 1", "printenv"]) {
      expect(rules.allow.some((r) => matchesRule(r, "Bash", cmd, "allow")), cmd).toBe(false);
    }
  });

  it("no bare Bash allow rule exists", () => {
    expect(rules.allow).not.toContain("Bash");
    expect(rules.allow.some((r) => r === "Bash(*)" || r === "Bash(:*)")).toBe(false);
  });

  it("denies reading or editing secrets and credentials", () => {
    expect(rules.deny).toEqual(
      expect.arrayContaining(["Read(//**/.secrets/**)", "Edit(//**/.secrets/**)", "Read(//home/**/.ssh/**)", "Read(//home/**/.claude/.credentials*)", "Read(//home/**/.config/gh/**)"]),
    );
  });
});

describe("minimalEnv", () => {
  it("keeps only the allow-listed variables", () => {
    const env = minimalEnv({
      PATH: "/bin", HOME: "/h", USER: "u", LANG: "C", TMPDIR: "/t", SHELL: "/bin/bash",
      WANDB_API_KEY: "secret", DASH_TOKEN: "t", GH_TOKEN: "g", ANTHROPIC_API_KEY: "a", AWS_SECRET_ACCESS_KEY: "s", SSH_AUTH_SOCK: "/s",
    });
    expect(Object.keys(env).sort()).toEqual(["HOME", "LANG", "PATH", "SHELL", "TMPDIR", "USER"]);
  });

  it("omits variables that are unset", () => {
    expect(minimalEnv({ PATH: "/bin" })).toEqual({ PATH: "/bin" });
  });
});

describe("workerOptions", () => {
  const o = workerOptions(cfg, { cwd: "/r/agents/a", env: { PATH: "/bin", HOME: "/h", WANDB_API_KEY: "x" } });

  it("isolates the session from user settings, plugins, hooks and connectors", () => {
    expect(o.settingSources).toEqual([]);
    expect(o.strictMcpConfig).toBe(true);
    expect(o.extraArgs).toEqual({ "safe-mode": null });
    expect(o.permissionMode).toBe("default");
  });

  it("uses the configured model, turn cap and a minimal environment", () => {
    expect(o.model).toBe("sonnet");
    expect(o.maxTurns).toBe(cfg.agentMaxTurns);
    expect(o.cwd).toBe("/r/agents/a");
    expect(o.env).toEqual({ PATH: "/bin", HOME: "/h" });
  });

  it("carries the permission rules inline", () => {
    expect(o.settings.permissions).toEqual(permissionRules());
  });

  it("lets a caller override the model", () => {
    expect(workerOptions(cfg, { cwd: "/w", env: {}, model: "haiku" }).model).toBe("haiku");
  });
});

describe("chiefOptions", () => {
  const o = chiefOptions(cfg, { cwd: "/r/repo", env: { PATH: "/bin" } });

  it("is read-only and has no shell; only listing workers is free, every other dashboard tool asks", () => {
    expect(o.settings.permissions.allow).toEqual(["Read", "Grep", "Glob", "mcp__dashboard__list_workers"]);
    expect(o.settings.permissions.ask).toEqual(
      ["spawn_worker", "message_worker", "stop_worker", "create_task", "update_task"].map((t) => `mcp__dashboard__${t}`),
    );
    expect(o.settings.permissions.deny).toEqual(expect.arrayContaining(["Bash", "Edit", "Write", "NotebookEdit"]));
    expect(o.model).toBe("opus");
  });

  it("shares the isolation settings", () => {
    expect(o.settingSources).toEqual([]);
    expect(o.strictMcpConfig).toBe(true);
    expect(o.extraArgs).toEqual({ "safe-mode": null });
  });
});
