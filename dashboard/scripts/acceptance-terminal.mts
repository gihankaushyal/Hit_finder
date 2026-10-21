/**
 * Real terminal acceptance for milestone 3. Starts the dashboard on a spare port against a throwaway repo and a
 * PRIVATE tmux socket (your own tmux server is never touched), drives the terminal over the HTTP API, and checks
 * output, `claude --version`, resize, the stream limit, persistence across a close and across a dashboard restart,
 * and that no helper process is left behind.   npx tsx scripts/acceptance-terminal.mts
 */
import { execFileSync, spawn, spawnSync, type ChildProcess } from "node:child_process";
import fs from "node:fs";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const DASH = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const root = fs.mkdtempSync(path.join(os.tmpdir(), "dash-term-accept-"));
const repo = path.join(root, "repo");
const stateDir = path.join(root, "state");
const socket = `dash-accept-${process.pid}`;
const results: [string, boolean, string][] = [];
const check = (name: string, ok: boolean, detail = "") => { results.push([name, ok, detail]); console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? `  (${detail})` : ""}`); };
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const bridges = () => spawnSync("bash", ["-c", `ps -eo args -u $USER | grep -c '[p]tyBridge.py.*${socket}'; true`], { encoding: "utf8" }).stdout.trim();

fs.mkdirSync(repo, { recursive: true });
execFileSync("git", ["init", "-q", "-b", "main"], { cwd: repo });
fs.writeFileSync(path.join(repo, "phase-05-kanban.md"), "");
fs.mkdirSync(stateDir, { recursive: true, mode: 0o700 });
const token = "c".repeat(64);
fs.writeFileSync(path.join(stateDir, "token"), token, { mode: 0o600 });
const port = await new Promise<number>((res) => { const s = net.createServer(); s.listen(0, "127.0.0.1", () => { const p = (s.address() as net.AddressInfo).port; s.close(() => res(p)); }); });
const env: Record<string, string> = {
  ...(process.env as Record<string, string>), NODE_ENV: "production", DASH_PORT: String(port), DASH_REPO_ROOT: repo, DASH_STATE_DIR: stateDir, DASH_KANBAN: path.join(repo, "phase-05-kanban.md"),
  DASH_CHECKPOINTS: path.join(repo, "checkpoints"), DASH_GH_BIN: path.join(DASH, "e2e/fake-gh.mjs"), DASH_PYTHON: path.join(DASH, "e2e/fake-pytest.mjs"),
  FAKE_GH_STATE: path.join(root, "gh.json"), FAKE_GH_CALLS: path.join(root, "gh-calls"), FAKE_GH_SEED: "empty", DASH_AGENT_DRIVER: "fake", DASH_AGENTS_DIR: path.join(root, "agents"),
  DASH_TERMINAL_TMUX_SOCKET: socket, DASH_TERMINAL_SESSION: "accept",
};
let output = "";
let server: ChildProcess;
const start = () => {
  server = spawn(path.join(DASH, "node_modules/.bin/tsx"), ["server/main.ts"], { cwd: DASH, env, stdio: ["ignore", "pipe", "pipe"] });
  server.stdout?.on("data", (d: Buffer) => (output += d)); server.stderr?.on("data", (d: Buffer) => (output += d));
};
const base = `http://127.0.0.1:${port}`;
const H = { Authorization: `Bearer ${token}`, "Content-Type": "application/json" };
const api = async (method: string, url: string, body?: unknown) => {
  const r = await fetch(base + url, { method, headers: H, body: body === undefined ? undefined : JSON.stringify(body) });
  return { status: r.status, json: (await r.json().catch(() => ({}))) as any };
};
async function waitFor<T>(label: string, f: () => Promise<T | undefined | false>, ms = 30_000): Promise<T> {
  const end = Date.now() + ms;
  for (;;) { const v = await f(); if (v) return v as T; if (Date.now() > end) throw new Error(`timed out waiting for ${label}`); await sleep(200); }
}
const ready = () => waitFor("server", async () => { try { return (await api("GET", "/api/health")).status === 200; } catch { return false; } });

/** Opens and attaches a stream; text accumulates in `get()`. */
async function attach(cols = 100, rows = 30) {
  const id = (await api("POST", "/api/terminal/open", { cols, rows })).json.id as string;
  const ac = new AbortController();
  const res = await fetch(`${base}/api/terminal/${id}/stream`, { headers: H, signal: ac.signal });
  let text = "";
  const reader = res.body!.getReader();
  const pump = (async () => { try { for (;;) { const { done, value } = await reader.read(); if (done) break; text += Buffer.from(value).toString("utf8"); } } catch { /* aborted */ } })();
  const until = (re: RegExp, ms = 30_000) => waitFor(String(re), async () => (re.test(text) ? text : false), ms);
  return { id, status: res.status, get: () => text, until, send: (data: string) => api("POST", `/api/terminal/${id}/input`, { data }), close: async () => { await api("DELETE", `/api/terminal/${id}`); ac.abort(); await pump; } };
}

try {
  start();
  await ready();
  const a = await attach();
  check("stream attached", a.status === 200);
  await a.until(/\$ $/);
  check("a prompt appears (your login shell started inside tmux)", true);
  await a.send("echo answer-$((40+2))\r");
  await a.until(/\r\nanswer-42/);
  check("a command runs and its output comes back", true);

  await a.send("claude --version\r");
  const v = await a.until(/\d+\.\d+\.\d+ \(Claude Code\)/).catch(() => "");
  check("claude runs inside the terminal", /\(Claude Code\)/.test(String(v)), /(\d+\.\d+\.\d+ \(Claude Code\))/.exec(String(v))?.[1]);

  check("resize accepted", (await api("POST", `/api/terminal/${a.id}/resize`, { cols: 120, rows: 40 })).status === 200);
  await a.send("stty size\r");
  await a.until(/40 120/);
  check("the shell sees the new size", true);

  const b = await attach();
  const third = await api("POST", "/api/terminal/open", { cols: 80, rows: 24 });
  check("a third stream is refused at the limit", third.status === 409, `HTTP ${third.status}`);
  await b.close();
  await a.close();
  await sleep(1500);
  check("no helper process left after the streams are closed", bridges() === "0", `count=${bridges()}`);

  const again = await attach();
  await again.until(/answer-42/);
  check("a new stream finds the same tmux session and its history", true);
  await again.close();

  server.kill("SIGTERM");
  await sleep(2500);
  check("dashboard shutdown leaves no helper process", bridges() === "0", `count=${bridges()}`);
  start();
  await ready();
  const after = await attach();
  await after.until(/answer-42/);
  check("after a dashboard restart the session is still there", true);
  await after.close();
} catch (e) {
  check("run completed", false, e instanceof Error ? e.message : String(e));
  console.log("--- server output ---\n" + output.slice(-3000));
} finally {
  server!.kill("SIGTERM");
  await sleep(2500);
  if (server!.exitCode === null) server!.kill("SIGKILL");
  spawnSync("tmux", ["-L", socket, "kill-server"]);
  check("no helper process left at the end", bridges() === "0", `count=${bridges()}`);
  check("the private tmux server is gone", spawnSync("tmux", ["-L", socket, "ls"], { encoding: "utf8" }).status !== 0);
  fs.rmSync(root, { recursive: true, force: true });
  const failed = results.filter(([, ok]) => !ok).length;
  console.log(`\n${results.length - failed}/${results.length} checks passed`);
  process.exit(failed ? 1 : 0);
}
