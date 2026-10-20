/**
 * Real-worker acceptance for milestone 2. Starts the dashboard on a spare port against a throwaway repo
 * and bare remote under the home directory, runs one real Haiku worker through the HTTP API, and checks
 * that a push waits on a card, that deny leaves the remote unchanged and that approve pushes.
 * It never touches Hit_finder, its jobs, its checkpoints or any secret.   npx tsx scripts/acceptance-agents.mts
 */
import { execFileSync, spawn, spawnSync, type ChildProcess } from "node:child_process";
import fs from "node:fs";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const DASH = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const root = fs.mkdtempSync(path.join(os.homedir(), ".dash-accept-"));
const origin = path.join(root, "origin.git");
const repo = path.join(root, "repo");
const agentsDir = path.join(root, "agents");
const stateDir = path.join(root, "state");
const results: [string, boolean, string][] = [];
const check = (name: string, ok: boolean, detail = "") => { results.push([name, ok, detail]); console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? `  (${detail})` : ""}`); };
const git = (cwd: string, ...a: string[]) => execFileSync("git", a, { cwd, encoding: "utf8" }).trim();
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

execFileSync("git", ["init", "-q", "--bare", "-b", "main", origin]);
execFileSync("git", ["clone", "-q", origin, repo], { stdio: "ignore" });
git(repo, "config", "user.name", "accept"); git(repo, "config", "user.email", "accept@example.invalid");
fs.writeFileSync(path.join(repo, "README.md"), "acceptance\n");
git(repo, "add", "."); git(repo, "commit", "-q", "-m", "init"); git(repo, "branch", "-M", "main"); git(repo, "push", "-q", "origin", "main");
fs.writeFileSync(path.join(repo, ".git", "info", "exclude"), "phase-05-kanban.md\n");
fs.writeFileSync(path.join(repo, "phase-05-kanban.md"), "");
const mainState = () => [git(repo, "rev-parse", "HEAD"), git(repo, "branch", "--show-current"), git(repo, "status", "--porcelain")].join("|");
const mainBefore = mainState();
const remoteHeads = () => git(repo, "ls-remote", "--heads", origin);
const headsBefore = remoteHeads();

const port = await new Promise<number>((res) => { const s = net.createServer(); s.listen(0, "127.0.0.1", () => { const p = (s.address() as net.AddressInfo).port; s.close(() => res(p)); }); });
const token = "b".repeat(64);
fs.mkdirSync(stateDir, { recursive: true, mode: 0o700 });
fs.writeFileSync(path.join(stateDir, "token"), token, { mode: 0o600 });
const env: Record<string, string> = {
  PATH: process.env.PATH ?? "", HOME: os.homedir(), NODE_ENV: "production",
  DASH_PORT: String(port), DASH_REPO_ROOT: repo, DASH_STATE_DIR: stateDir, DASH_AGENTS_DIR: agentsDir, DASH_KANBAN: path.join(repo, "phase-05-kanban.md"),
  DASH_CHECKPOINTS: path.join(repo, "checkpoints"), DASH_GH_BIN: path.join(DASH, "e2e/fake-gh.mjs"), DASH_PYTHON: path.join(DASH, "e2e/fake-pytest.mjs"),
  FAKE_GH_STATE: path.join(root, "gh.json"), FAKE_GH_CALLS: path.join(root, "gh-calls"), FAKE_GH_SEED: "empty",
  DASH_AGENT_DRIVER: "sdk", DASH_WORKER_MODEL: "haiku", DASH_AGENT_BASE: "main", DASH_AGENT_MAX_TURNS: "12",
};
let output = "";
const server: ChildProcess = spawn(path.join(DASH, "node_modules/.bin/tsx"), ["server/main.ts"], { cwd: DASH, env, stdio: ["ignore", "pipe", "pipe"] });
server.stdout?.on("data", (d: Buffer) => (output += d)); server.stderr?.on("data", (d: Buffer) => (output += d));
const base = `http://127.0.0.1:${port}`;
const api = async (method: string, url: string, body?: unknown) => {
  const r = await fetch(base + url, { method, headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" }, body: body === undefined ? undefined : JSON.stringify(body) });
  return { status: r.status, json: (await r.json().catch(() => ({}))) as any };
};
async function waitFor<T>(label: string, f: () => Promise<T | undefined | false>, ms = 150_000): Promise<T> {
  const end = Date.now() + ms;
  for (;;) { const v = await f(); if (v) return v as T; if (Date.now() > end) throw new Error(`timed out waiting for ${label}`); await sleep(1000); }
}
const agentState = async (name: string) => (await api("GET", "/api/agents")).json.agents.find((a: any) => a.name === name)?.state as string | undefined;

try {
  await waitFor("server", async () => { try { return (await api("GET", "/api/health")).status === 200; } catch { return false; } }, 30_000);
  const created = await api("POST", "/api/agents", { name: "accept-push", task: "In the current directory create a file hello.txt containing the text hi. Run `git add hello.txt`, then `git commit -m 'add hello'`, then `git push origin HEAD`. Report each result in one short line. Do not run any other commands." });
  check("agent created", created.status === 201, `HTTP ${created.status}`);

  const card1 = await waitFor("first approval card", async () => (await api("GET", "/api/approvals")).json.approvals[0]);
  check("push waits on a card", /git push/.test(card1.input.command ?? ""), card1.input.command);
  check("agent shows waiting-approval", (await agentState("accept-push")) === "waiting-approval");
  await sleep(4000);
  check("remote unchanged while the card is pending", remoteHeads() === headsBefore);

  const denied = await api("POST", `/api/approvals/${card1.id}/decision`, { decision: "deny", message: "Not now" });
  check("deny accepted", denied.status === 200);
  await waitFor("agent idle after deny", async () => (await agentState("accept-push")) === "idle");
  check("remote unchanged after deny", remoteHeads() === headsBefore);
  const wt = path.join(agentsDir, "accept-push");
  check("commit exists in the worktree", git(wt, "log", "-1", "--format=%s").startsWith("add hello"), git(wt, "log", "-1", "--format=%s").slice(0, 60));

  await api("POST", "/api/agents/accept-push/message", { text: "Try `git push origin HEAD` once more." });
  const card2 = await waitFor("second approval card", async () => (await api("GET", "/api/approvals")).json.approvals[0]);
  check("second push raises a new card", card2.id !== card1.id);
  const allowed = await api("POST", `/api/approvals/${card2.id}/decision`, { decision: "allow" });
  check("approve accepted", allowed.status === 200);
  await waitFor("agent idle after approve", async () => (await agentState("accept-push")) === "idle");
  check("remote has the agent branch after approve", remoteHeads().includes("refs/heads/phase-05-agent-accept-push"), remoteHeads().split("\n").length + " heads");

  const log = (await api("GET", "/api/agents/accept-push/events?after=0&limit=500")).json.events as any[];
  check("log records approvals", log.filter((e) => e.kind === "approval").map((e) => e.outcome).join(",").includes("requested"), log.filter((e) => e.kind === "approval").map((e) => e.outcome).join(","));
  check("no secret or home path leaked into the log", !JSON.stringify(log).includes(os.homedir() + "/.ssh"));
  check("main repo untouched", mainState() === mainBefore);

  await api("POST", "/api/agents/accept-push/stop");
  await sleep(1500);
  const leftover = spawnSync("bash", ["-c", `for p in $(pgrep -u $USER -x 'claude|bwrap|socat'); do readlink /proc/$p/cwd 2>/dev/null; done | grep -c '${agentsDir}'`], { encoding: "utf8" }).stdout.trim();
  check("no agent processes left after stop", leftover === "0" || leftover === "", `count=${leftover || 0}`);
} catch (e) {
  check("run completed", false, e instanceof Error ? e.message : String(e));
  console.log("--- server output ---\n" + output.slice(-3000));
} finally {
  server.kill("SIGTERM");
  await sleep(2500);
  if (server.exitCode === null) server.kill("SIGKILL");
  const after = spawnSync("bash", ["-c", `for p in $(pgrep -u $USER -x 'claude|bwrap|socat'); do readlink /proc/$p/cwd 2>/dev/null; done | grep -c '${agentsDir}'`], { encoding: "utf8" }).stdout.trim();
  check("no agent processes left after server shutdown", after === "0" || after === "", `count=${after || 0}`);
  fs.rmSync(root, { recursive: true, force: true });
  const failed = results.filter(([, ok]) => !ok).length;
  console.log(`\n${results.length - failed}/${results.length} checks passed`);
  process.exit(failed ? 1 : 0);
}
