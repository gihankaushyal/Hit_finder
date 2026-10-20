import fs from "node:fs";
import { spawn } from "node:child_process";
import { query, type CanUseTool, type Options, type SpawnOptions } from "@anthropic-ai/claude-agent-sdk";
import { bwrapArgs, claudeDirGuards, guardedGitPaths } from "./bwrap";
import type { AgentDriver, DriverItem, Session, SessionStart } from "./driver";
import { normalise } from "./normalise";

/**
 * The only file that imports the Agent SDK. It drives the installed `claude` through the user's
 * claude.ai login (no API key), optionally inside bubblewrap, with streaming input so a session
 * stays open for follow-up messages.
 */

const STDERR_TAIL_BYTES = 4000;

function userMessage(text: string) {
  return { type: "user" as const, message: { role: "user" as const, content: text }, parent_tool_use_id: null, session_id: "" };
}

function spawner(s: SessionStart, onStderr: (chunk: string) => void) {
  const c = s.confinement;
  return (o: SpawnOptions) => {
    const env = o.env as NodeJS.ProcessEnv;
    let command = o.command;
    let args = o.args;
    if (c) {
      const guarded = [...(c.gitDir && c.ownBranch ? guardedGitPaths(c.gitDir, c.ownBranch) : []), ...claudeDirGuards(c.home), ...c.guarded];
      args = bwrapArgs({ ...c, guarded, maskDirs: c.maskDirs.filter((d) => fs.existsSync(d)), cwd: o.cwd ?? c.worktree, command: o.command, args: o.args });
      command = "bwrap";
    }
    // Detached so the whole process group can be killed on abort; bwrap's --die-with-parent covers a crash of ours.
    const child = spawn(command, args, { env, cwd: o.cwd, stdio: ["pipe", "pipe", "pipe"], detached: true });
    child.stderr?.on("data", (d: Buffer) => onStderr(d.toString()));
    o.signal.addEventListener("abort", () => {
      try {
        process.kill(-(child.pid as number), "SIGKILL");
      } catch {
        /* already gone */
      }
    });
    return child as unknown as ReturnType<NonNullable<Options["spawnClaudeCodeProcess"]>>;
  };
}

export function createSdkDriver(): AgentDriver {
  return {
    start(s: SessionStart): Session {
      const ac = new AbortController();
      const inbox: string[] = [s.prompt];
      let wake: (() => void) | null = null;
      let closed = false;
      let stderrTail = "";
      const onStderr = (chunk: string) => {
        stderrTail = (stderrTail + chunk).slice(-STDERR_TAIL_BYTES);
      };

      async function* input() {
        while (!closed) {
          const next = inbox.shift();
          if (next === undefined) {
            await new Promise<void>((r) => (wake = r));
            wake = null;
            continue;
          }
          yield userMessage(next);
        }
      }

      const canUseTool: CanUseTool = async (tool, toolInput, ctx) => {
        const out = await s.canUseTool(tool, toolInput, { signal: ctx.signal, reason: ctx.decisionReason ? String(ctx.decisionReason) : undefined });
        return out.decision === "allow" ? { behavior: "allow", updatedInput: toolInput } : { behavior: "deny", message: out.message ?? "Denied by the user." };
      };

      const o = s.options;
      const q = query({
        prompt: input(),
        options: {
          cwd: o.cwd,
          model: o.model,
          maxTurns: o.maxTurns,
          env: o.env,
          settingSources: o.settingSources,
          strictMcpConfig: o.strictMcpConfig,
          extraArgs: o.extraArgs,
          permissionMode: o.permissionMode,
          settings: { permissions: { ...o.settings.permissions } },
          pathToClaudeCodeExecutable: s.claudeBin,
          includePartialMessages: false,
          resume: s.resume,
          abortController: ac,
          canUseTool,
          spawnClaudeCodeProcess: spawner(s, onStderr),
        },
      });

      async function* events(): AsyncGenerator<DriverItem> {
        try {
          for await (const m of q) for (const item of normalise(m)) yield item;
        } catch (e) {
          if (closed) return;
          const tail = stderrTail.trim().split("\n").pop() ?? "";
          throw new Error(`Claude session failed: ${e instanceof Error ? e.message : String(e)}${tail ? ` (${tail.slice(0, 200)})` : ""}`);
        }
      }

      return {
        events: events(),
        send(text) {
          inbox.push(text);
          wake?.();
        },
        async interrupt() {
          await q.interrupt();
        },
        close() {
          if (closed) return;
          closed = true;
          wake?.();
          ac.abort();
          q.close?.();
        },
      };
    },
  };
}
