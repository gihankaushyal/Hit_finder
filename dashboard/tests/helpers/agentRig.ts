import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { vi } from "vitest";
import { buildServer } from "../../server/wiring";
import { createSupervisor, type Supervisor } from "../../server/agents/supervisor";
import { createFakeDriver, type FakeDriverOptions } from "../../server/agents/fakeDriver";
import { createApprovals, type Approvals } from "../../server/agents/approvals";
import { createChief, type Chief } from "../../server/agents/chief";
import { createChiefTools } from "../../server/agents/chiefTools";
import { createWorktrees } from "../../server/agents/worktrees";
import { EventBus } from "../../server/events";
import { loadConfig, type Config } from "../../server/config";
import type { Gh } from "../../server/lib/gh";

export const AGENT_TOKEN = "a".repeat(64);
export const AUTH = { Authorization: `Bearer ${AGENT_TOKEN}`, "Content-Type": "application/json", Origin: "http://127.0.0.1:4317", Host: "127.0.0.1:4317" };

export interface AgentRig {
  app: ReturnType<typeof buildServer>;
  sup: Supervisor;
  chief: Chief;
  approvals: Approvals;
  bus: EventBus;
  cfg: Config;
  root: string;
  repo: string;
  cleanup(): Promise<void>;
}

/** A real temp git repo, a real supervisor and the fake driver, mounted behind the real app. */
export async function makeAgentRig(env: Record<string, string> = {}, driver: FakeDriverOptions = { stepMs: 0 }): Promise<AgentRig> {
  const root = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), "dash-rig-")));
  const repo = path.join(root, "repo");
  fs.mkdirSync(repo);
  const git = (...a: string[]) => execFileSync("git", ["-C", repo, ...a], { encoding: "utf8" });
  git("init", "-q", "-b", "phase-05");
  git("config", "user.email", "t@example.com");
  git("config", "user.name", "T");
  fs.writeFileSync(path.join(repo, "a.txt"), "1");
  git("add", "a.txt");
  git("commit", "-q", "-m", "init");
  const cfg = loadConfig({ DASH_REPO_ROOT: repo, DASH_AGENTS_DIR: path.join(root, "agents"), DASH_STATE_DIR: path.join(root, "state"), DASH_AGENT_BASE: "phase-05", DASH_AGENT_DRIVER: "fake", ...env });
  const bus = new EventBus();
  const approvals = createApprovals({ bus, auditPath: path.join(root, "audit.jsonl") });
  const sup = createSupervisor({
    cfg,
    bus,
    driver: createFakeDriver(driver),
    approvals,
    worktrees: createWorktrees({ repoRoot: repo, agentsDir: cfg.agentsDir, base: cfg.agentBase }),
    env: { PATH: process.env.PATH },
    notifyMs: 5,
  });
  await sup.init();
  const kanbanStub = { board: vi.fn(), sync: vi.fn(), addTask: vi.fn(async () => {}), setStatus: vi.fn(async () => {}), fileChanged: vi.fn(), poll: vi.fn(), dispose: vi.fn() };
  const chief = createChief({
    cfg,
    bus,
    driver: createFakeDriver(driver),
    approvals,
    tools: createChiefTools({ supervisor: sup, kanban: kanbanStub as never }),
    env: { PATH: process.env.PATH },
    notifyMs: 5,
  });
  await chief.init();
  const gh = { json: vi.fn(), text: vi.fn() } as unknown as Gh;
  const app = buildServer({
    config: { ...cfg, webDist: "/nonexistent" },
    token: AGENT_TOKEN,
    bus,
    gh,
    tests: { state: () => ({ status: "idle" }) as never, start: () => true, stop: () => Promise.resolve() },
    kanban: kanbanStub as never,
    runner: async () => ({ stdout: "x\n", stderr: "", code: 0 }),
    runs: { scan: async () => ({ runs: [], scannedAt: "", checkpointsDirExists: false, truncated: false, error: null }) },
    agents: sup,
    approvals,
    chief,
  });
  return {
    app, sup, chief, approvals, bus, cfg, root, repo,
    async cleanup() {
      await chief.stop();
      await sup.stopAll(1000);
      fs.rmSync(root, { recursive: true, force: true });
    },
  };
}

export async function until<T>(f: () => T | undefined | false, ms = 3000): Promise<T> {
  const end = Date.now() + ms;
  for (;;) {
    const v = f();
    if (v) return v as T;
    if (Date.now() > end) throw new Error("timed out waiting");
    await new Promise((r) => setTimeout(r, 5));
  }
}
