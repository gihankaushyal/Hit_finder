/**
 * Real chief-of-staff acceptance for milestone 2b. Starts the dashboard on a spare port against a throwaway repo
 * and bare remote under the home directory, runs a real Haiku chief of staff that spawns a real Haiku worker through the HTTP API,
 * and checks that the spawn waits on a card, deny creates nothing, approve creates the worker, and list_workers is free.
 * It never touches Hit_finder, its jobs, its checkpoints or any secret.   npx tsx scripts/acceptance-chief.mts
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
  DASH_AGENT_DRIVER: "sdk", DASH_CHIEF_MODEL: "haiku", DASH_WORKER_MODEL: "haiku", DASH_AGENT_BASE: "main", DASH_AGENT_MAX_TURNS: "12",
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

const chiefState = async () => (await api("GET", "/api/chief")).json.state as string;
const cards = async () => (await api("GET", "/api/approvals")).json.approvals as any[];
const workers = async () => (await api("GET", "/api/agents")).json.agents as any[];
const transcript = async () => ((await api("GET", "/api/chief/events?after=0&limit=500")).json.events as any[]);

try {
  await waitFor("server", async () => { try { return (await api("GET", "/api/health")).status === 200; } catch { return false; } }, 30_000);
  const first = await api("POST", "/api/chief/message", { text: "Use the spawn_worker tool to start a worker named chat-docs. Its task: create a file notes.txt containing the word hello, then commit it with git. Do nothing else yourself." });
  check("message accepted", first.status === 200, `HTTP ${first.status}`);

  const card1 = await waitFor("spawn card", async () => (await cards())[0]);
  check("spawn waits on a card from the chief", card1.agentName === "Chief of staff" && card1.tool === "mcp__dashboard__spawn_worker", `${card1.tool} ${JSON.stringify(card1.input).slice(0, 100)}`);
  check("card shows the worker name", card1.input?.name === "chat-docs", String(card1.input?.name));
  check("chief shows waiting-approval", (await chiefState()) === "waiting-approval");
  await sleep(3000);
  check("no worker exists while the card is pending", (await workers()).length === 0);

  check("deny accepted", (await api("POST", `/api/approvals/${card1.id}/decision`, { decision: "deny", message: "Not yet" })).status === 200);
  await waitFor("chief idle after deny", async () => (await chiefState()) === "idle");
  check("no worker after deny", (await workers()).length === 0);

  await api("POST", "/api/chief/message", { text: "I changed my mind. Please spawn chat-docs now, with the same task." });
  const card2 = await waitFor("second spawn card", async () => (await cards())[0]);
  check("second spawn raises a new card", card2.id !== card1.id && card2.input?.name === "chat-docs");
  check("approve accepted", (await api("POST", `/api/approvals/${card2.id}/decision`, { decision: "allow" })).status === 200);
  await waitFor("worker created", async () => (await workers()).find((w: any) => w.name === "chat-docs"));
  check("worker exists after approve", true);
  await waitFor("worker finished its turn", async () => (await workers()).find((w: any) => w.name === "chat-docs")?.state === "idle", 180_000);
  await waitFor("chief idle after spawn", async () => (await chiefState()) === "idle");
  const wt = path.join(agentsDir, "chat-docs");
  check("worker committed notes.txt", fs.existsSync(path.join(wt, "notes.txt")) && git(wt, "log", "-1", "--format=%s").length > 0, git(wt, "log", "-1", "--format=%s").slice(0, 50));

  await api("POST", "/api/chief/message", { text: "Use the list_workers tool and tell me the names you see." });
  await waitFor("list reply", async () => (await chiefState()) === "idle" && (await transcript()).filter((e) => e.kind === "user").length === 3);
  const before = (await transcript());
  check("list_workers raised no card", (await cards()).length === 0 && !before.slice(-12).some((e) => e.kind === "approval" && e.tool === "mcp__dashboard__list_workers"));
  check("chief called list_workers", before.some((e) => e.kind === "tool-use" && e.tool === "mcp__dashboard__list_workers"));

  const t = JSON.stringify(before);
  check("no home path or token in the chat log", !t.includes(os.homedir() + "/.ssh") && !t.includes(token));
  check("main repo untouched", mainState() === mainBefore);
  await api("POST", "/api/agents/chat-docs/stop");
  await api("POST", "/api/chief/stop");
  await sleep(1500);
  const leftover = spawnSync("bash", ["-c", `for p in $(pgrep -u $USER -x 'claude|bwrap|socat'); do readlink /proc/$p/cwd 2>/dev/null; done | grep -c -e '${agentsDir}' -e '${repo}'`], { encoding: "utf8" }).stdout.trim();
  check("no agent processes left after stop", leftover === "0" || leftover === "", `count=${leftover || 0}`);
} catch (e) {
  check("run completed", false, e instanceof Error ? e.message : String(e));
  console.log("--- server output ---\n" + output.slice(-3000));
} finally {
  server.kill("SIGTERM");
  await sleep(2500);
  if (server.exitCode === null) server.kill("SIGKILL");
  const after = spawnSync("bash", ["-c", `for p in $(pgrep -u $USER -x 'claude|bwrap|socat'); do readlink /proc/$p/cwd 2>/dev/null; done | grep -c -e '${agentsDir}' -e '${repo}'`], { encoding: "utf8" }).stdout.trim();
  check("no agent processes left after server shutdown", after === "0" || after === "", `count=${after || 0}`);
  fs.rmSync(root, { recursive: true, force: true });
  const failed = results.filter(([, ok]) => !ok).length;
  console.log(`\n${results.length - failed}/${results.length} checks passed`);
  process.exit(failed ? 1 : 0);
}
