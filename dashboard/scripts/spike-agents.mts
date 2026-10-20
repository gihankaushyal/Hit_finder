/**
 * Milestone 2 spike. Real but tiny and throwaway: each check starts one or two headless Claude Code
 * sessions with the cheapest model inside a temporary git repo under the OS temp dir. It never touches
 * Hit_finder, its jobs, its checkpoints or any secret. Not part of the build or the test suite.
 *
 *   npx tsx scripts/spike-agents.mts <check> [--sandbox] [--decision allow|deny] [--keep]
 *   checks: login | push | sandbox | hooks | resume | concurrent | interrupt
 *
 * Every message of a check is written to <tmp>/<check>.jsonl so the results can be turned into fixtures.
 */
import { execFileSync, spawn, spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { query, type CanUseTool, type Options, type SDKMessage, type SpawnOptions } from "@anthropic-ai/claude-agent-sdk";

const args = process.argv.slice(2);
const check = args[0];
const flag = (n: string) => args.includes(`--${n}`);
const opt = (n: string, d: string) => (args.includes(`--${n}`) ? args[args.indexOf(`--${n}`) + 1] : d);
const SANDBOX = flag("sandbox");
const DECISION = opt("decision", "deny") as "allow" | "deny";
const HOLD_MS = Number(opt("hold-ms", "4000"));
const MODEL = opt("model", "haiku");
const WATCHDOG_MS = Number(opt("watchdog-ms", "100000"));
const DEBUG = flag("debug");
const BWRAP = flag("bwrap");
const SAFE = flag("safe");

const CLAUDE_BIN = execFileSync("which", ["claude"], { encoding: "utf8" }).trim();
const git = (cwd: string, ...a: string[]) => execFileSync("git", a, { cwd, encoding: "utf8" }).trim();
const sh = (cwd: string, cmd: string) => spawnSync("bash", ["-c", cmd], { cwd, encoding: "utf8" });

const ENV: Record<string, string> = {};
for (const k of ["PATH", "HOME", "USER", "LANG", "TMPDIR", "SHELL"]) if (process.env[k]) ENV[k] = process.env[k] as string;

interface Rig { root: string; origin: string; repo: string; wt: (name: string) => string }

function makeRig(): Rig {
  // With --bwrap the rig lives in the home directory, which the worker sees read-only except its own worktree
  // (like the real main checkout); otherwise in the OS temp dir.
  const root = BWRAP ? fs.mkdtempSync(path.join(os.homedir(), ".m2spike-")) : fs.mkdtempSync(path.join(os.tmpdir(), "m2spike-"));
  const origin = path.join(root, "origin.git");
  const repo = path.join(root, "repo");
  execFileSync("git", ["init", "-q", "--bare", origin]);
  execFileSync("git", ["clone", "-q", origin, repo]);
  git(repo, "config", "user.name", "spike");
  git(repo, "config", "user.email", "spike@example.invalid");
  fs.writeFileSync(path.join(repo, "README.md"), "spike\n");
  fs.writeFileSync(path.join(repo, "package.json"), JSON.stringify({ name: "spike", scripts: { test: "node -e \"process.exit(0)\"" } }));
  git(repo, "add", ".");
  git(repo, "commit", "-q", "-m", "init");
  git(repo, "push", "-q", "origin", "HEAD:refs/heads/main");
  fs.mkdirSync(path.join(root, "outside"));
  const wt = (name: string) => {
    const p = path.join(root, `wt-${name}`);
    if (!fs.existsSync(p)) git(repo, "worktree", "add", "-q", p, "-b", `phase-05-agent-${name}`);
    return p;
  };
  return { root, origin, repo, wt };
}

const short = (v: unknown, n = 220) => { const s = typeof v === "string" ? v : (JSON.stringify(v) ?? "(none)"); return s.length > n ? `${s.slice(0, n)}...` : s; };

function describe(m: SDKMessage, counts: Record<string, number>): void {
  counts[m.type] = (counts[m.type] ?? 0) + 1;
  const any = m as any;
  if (m.type === "system" && any.subtype === "init") {
    console.log(`  [init] model=${any.model} cwd=${any.cwd} permissionMode=${any.permissionMode} apiKeySource=${any.apiKeySource} tools=${any.tools?.length} session=${any.session_id}`);
  } else if (m.type === "assistant") {
    for (const b of any.message?.content ?? []) {
      if (b.type === "text") console.log(`  [assistant] ${short(b.text)}`);
      if (b.type === "tool_use") console.log(`  [tool_use] ${b.name} ${short(b.input)}`);
    }
  } else if (m.type === "user") {
    for (const b of any.message?.content ?? []) if (b?.type === "tool_result") console.log(`  [tool_result] ${short(b.content, 160)}`);
  } else if (m.type === "result") {
    console.log(`  [result] ${any.subtype} is_error=${any.is_error} turns=${any.num_turns} cost=$${any.total_cost_usd?.toFixed?.(4)} ms=${any.duration_ms} denials=${any.permission_denials?.length ?? 0} :: ${short(any.result, 200)}`);
  } else if (m.type !== "stream_event") {
    console.log(`  [${m.type}${any.subtype ? "/" + any.subtype : ""}]`);
  }
}

/**
 * Runs the whole worker inside bubblewrap: the filesystem is read-only except the worktree, the repo's .git
 * (with hooks, config, HEAD and info overlaid read-only), temp and ~/.claude; .secrets is masked.
 */
function bwrapSpawner(rig: Rig, wt: string) {
  return (o: SpawnOptions) => {
    const home = os.homedir();
    const gitDir = path.join(rig.repo, ".git");
    const a = ["--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--unshare-pid", "--die-with-parent", "--new-session",
      "--bind", wt, wt, "--bind", gitDir, gitDir, "--bind", rig.origin, rig.origin, "--tmpfs", "/tmp", "--bind", path.join(home, ".claude"), path.join(home, ".claude")];
    for (const guarded of ["hooks", "config", "HEAD", "info", "packed-refs"]) {
      const g = path.join(gitDir, guarded);
      if (fs.existsSync(g)) a.push("--ro-bind", g, g);
    }
    a.push("--chdir", o.cwd ?? wt, "--", o.command, ...o.args);
    const child = spawn("bwrap", a, { env: o.env as NodeJS.ProcessEnv, stdio: ["pipe", "pipe", "pipe"], detached: true });
    child.stderr.on("data", (d: Buffer) => fs.appendFileSync(path.join(rig.root, "bwrap.stderr.log"), d));
    o.signal.addEventListener("abort", () => { try { process.kill(-(child.pid as number), "SIGKILL"); } catch { /* already gone */ } });
    return child as any;
  };
}

interface RunResult { messages: SDKMessage[]; sessionId?: string; result?: any; counts: Record<string, number>; asks: { tool: string; input: any; at: number; decidedAt?: number; decision?: string }[] }

async function run(rig: Rig, label: string, prompt: string | AsyncIterable<any>, options: Options, decide?: (tool: string, input: any) => Promise<"allow" | "deny">): Promise<RunResult> {
  const log = fs.createWriteStream(path.join(rig.root, `${label}.jsonl`), { flags: "a" });
  const out: RunResult = { messages: [], counts: {}, asks: [] };
  const canUseTool: CanUseTool = async (tool, input) => {
    const rec: RunResult["asks"][number] = { tool, input, at: Date.now() };
    out.asks.push(rec);
    console.log(`  [canUseTool] ASK ${tool} ${short(input, 160)}`);
    const d = decide ? await decide(tool, input) : "deny";
    rec.decidedAt = Date.now();
    rec.decision = d;
    console.log(`  [canUseTool] -> ${d} after ${rec.decidedAt - rec.at} ms`);
    return d === "allow" ? { behavior: "allow", updatedInput: input } : { behavior: "deny", message: "Denied by the spike" };
  };
  // A stall must never leave a claude child behind: the watchdog aborts the query, which ends the child.
  const ac = options.abortController ?? new AbortController();
  const watchdog = setTimeout(() => { console.log(`  WATCHDOG: no result after ${WATCHDOG_MS} ms, aborting`); ac.abort(); }, WATCHDOG_MS);
  const extra: Options = { ...(BWRAP ? { spawnClaudeCodeProcess: bwrapSpawner(rig, String(options.cwd)) } : {}), ...(SAFE ? { extraArgs: { "safe-mode": null }, strictMcpConfig: true } : {}), abortController: ac, stderr: (d: string) => fs.appendFileSync(path.join(rig.root, `${label}.stderr.log`), d), ...(DEBUG ? { debug: true, debugFile: path.join(rig.root, `${label}.debug.log`) } : {}) };
  const q = query({ prompt, options: { model: MODEL, pathToClaudeCodeExecutable: CLAUDE_BIN, settingSources: [], env: ENV, includePartialMessages: true, canUseTool, ...options, ...extra } });
  try { for await (const m of q) {
    out.messages.push(m);
    log.write(JSON.stringify(m) + "\n");
    describe(m, out.counts);
    if ((m as any).session_id) out.sessionId = (m as any).session_id;
    if (m.type === "result") out.result = m;
  } } catch (e: any) { console.log(`  query ended with: ${e?.name ?? e}`); } finally { clearTimeout(watchdog); q.close?.(); }
  log.end();
  console.log(`  message counts: ${JSON.stringify(out.counts)}`);
  return out;
}

/** Permission rules shared by the checks: broad git allow, a more specific ask for push, hard denies for anything outside the rig. */
function rules(rig: Rig, extraAllow: string[] = []) {
  const wt = rig.wt("a");
  const settings: any = {
    permissions: {
      allow: ["Bash(git *)", "Write", "Edit", "Read", ...extraAllow],
      ask: ["Bash(git push *)", "Bash(git push)"],
      deny: ["Bash(git -C *)", "Read(//data/bioxfel/**)", "Edit(//data/bioxfel/**)", "Write(//data/bioxfel/**)", "Read(//home/**/.ssh/**)"],
    },
  };
  if (SANDBOX) {
    settings.sandbox = {
      enabled: true,
      failIfUnavailable: true,
      filesystem: { allowWrite: [wt, path.join(rig.repo, ".git"), rig.origin, ...(flag("allow-tmp") ? ["/tmp"] : [])], denyRead: [path.join(os.homedir(), ".ssh"), path.join(os.homedir(), ".secrets")] },
    };
  }
  return { wt, settings };
}

async function main() {
  console.log(`claude binary: ${CLAUDE_BIN}  | sandbox=${SANDBOX} model=${MODEL} | API key in env: ${process.env.ANTHROPIC_API_KEY ? "YES (will not be passed)" : "no"}`);
  const rig = makeRig();
  console.log(`rig: ${rig.root}`);
  try {
    if (check === "login") {
      await run(rig, "login", "Reply with the single word OK.", { cwd: rig.wt("a"), maxTurns: 1 });
    } else if (check === "push") {
      const { wt, settings } = rules(rig);
      const before = git(rig.repo, "ls-remote", "--heads", rig.origin);
      const r = await run(rig, `push-${DECISION}${SANDBOX ? "-sandbox" : ""}`,
        "In the current directory create hello.txt containing the text hi, run `git add hello.txt`, run `git commit -m 'add hello'`, then run `git push origin HEAD`. Report each result in one short line. Do not run any other commands.",
        { cwd: wt, maxTurns: 8, settings },
        async (tool, input) => { if (tool === "Bash" && String(input.command).includes("git push")) { await new Promise((res) => setTimeout(res, HOLD_MS)); return DECISION; } return "deny"; });
      const after = git(rig.repo, "ls-remote", "--heads", rig.origin);
      console.log(`\nRESULT push: asks=${r.asks.length} unexpectedAsks=${r.asks.filter((a) => !(a.tool === "Bash" && String(a.input.command).includes("git push"))).length}`);
      console.log(`  origin heads before:\n${before.split("\n").map((l) => "    " + l).join("\n")}\n  after:\n${after.split("\n").map((l) => "    " + l).join("\n")}`);
      console.log(`  hello.txt exists in worktree: ${fs.existsSync(path.join(wt, "hello.txt"))}; origin has the agent branch: ${after.includes("phase-05-agent-a")}`);
    } else if (check === "sandbox") {
      const { wt, settings } = rules(rig, ["Bash"]);
      delete settings.sandbox; // the SDK sandbox hangs on this node; confinement comes from --bwrap
      const outside = path.join(rig.root, "outside");
      await run(rig, `sandbox${SANDBOX ? "-on" : "-off"}`,
        `Run these shell commands one at a time and report each exit code in one line: (1) echo outside > ${outside}/canary.txt (2) echo inside > ./inside.txt (3) git add inside.txt && git commit -m inside (4) npm test --silent (5) python3 -c "print(1)" (6) echo hook > ${rig.repo}/.git/hooks/pre-commit (7) echo x >> ${rig.repo}/README.md (8) mkdir -p ${path.join(os.homedir(), ".m2-spike-should-not-exist")}`,
        { cwd: wt, maxTurns: 10, settings });
      console.log(`\nRESULT sandbox: hook injected=${fs.existsSync(path.join(rig.repo, ".git", "hooks", "pre-commit"))} main-repo README modified=${fs.readFileSync(path.join(rig.repo, "README.md"), "utf8") !== "spike\n"} home dir created=${fs.existsSync(path.join(os.homedir(), ".m2-spike-should-not-exist"))};\n  outside canary written=${fs.existsSync(path.join(outside, "canary.txt"))}; inside.txt written=${fs.existsSync(path.join(wt, "inside.txt"))}; commits in worktree=${git(wt, "rev-list", "--count", "HEAD")}`);
    } else if (check === "hooks") {
      const wt = rig.wt("a");
      const canaryProject = path.join(wt, ".hook-canary");
      fs.mkdirSync(path.join(wt, ".claude"), { recursive: true });
      fs.writeFileSync(path.join(wt, ".claude", "settings.json"), JSON.stringify({ hooks: { SessionStart: [{ hooks: [{ type: "command", command: `touch ${canaryProject}` }] }] } }));
      await run(rig, "hooks-none", "Reply with the single word OK.", { cwd: wt, maxTurns: 1, settingSources: [] });
      console.log(`  project hook fired with settingSources []: ${fs.existsSync(canaryProject)}`);
      fs.rmSync(canaryProject, { force: true });
      await run(rig, "hooks-project", "Reply with the single word OK.", { cwd: wt, maxTurns: 1, settingSources: ["project"] });
      console.log(`  project hook fired with settingSources ["project"] (control): ${fs.existsSync(canaryProject)}`);
    } else if (check === "resume") {
      const wt = rig.wt("a");
      const first = await run(rig, "resume-1", "Remember the code word BANANA42. Reply with just OK.", { cwd: wt, maxTurns: 1 });
      const second = await run(rig, "resume-2", "What was the code word? Reply with just the word.", { cwd: wt, maxTurns: 1, resume: first.sessionId });
      console.log(`\nRESULT resume after a clean end: ${String(second.result?.result).includes("BANANA42") ? "remembered" : "NOT remembered"}`);
      const ac = new AbortController();
      const third = run(rig, "resume-3", "Count slowly from 1 to 300, one number per line, in words.", { cwd: wt, maxTurns: 2, abortController: ac, resume: first.sessionId }).catch((e) => console.log(`  aborted: ${e?.name ?? e}`));
      await new Promise((res) => setTimeout(res, 3500));
      ac.abort();
      await third;
      const fourth = await run(rig, "resume-4", "What was the code word? Reply with just the word.", { cwd: wt, maxTurns: 1, resume: first.sessionId });
      console.log(`\nRESULT resume after an aborted turn: ${String(fourth.result?.result).includes("BANANA42") ? "remembered" : "NOT remembered"}`);
    } else if (check === "concurrent") {
      const peak = { rssMb: 0, procs: 0 };
      const timer = setInterval(() => {
        const ps = sh(rig.root, `ps -u ${os.userInfo().username} -o rss=,args= | grep -- '--output-format stream-json' | grep -v grep`);
        const rows = ps.stdout.split("\n").filter(Boolean);
        const mb = rows.reduce((s, r) => s + Number(r.trim().split(/\s+/)[0]), 0) / 1024;
        if (mb > peak.rssMb) { peak.rssMb = mb; peak.procs = rows.length; }
      }, 300);
      const [a, b] = await Promise.all([
        run(rig, "concurrent-a", "Reply with just the word ALPHA.", { cwd: rig.wt("a"), maxTurns: 1 }),
        run(rig, "concurrent-b", "Reply with just the word BRAVO.", { cwd: rig.wt("b"), maxTurns: 1 }),
      ]);
      clearInterval(timer);
      console.log(`\nRESULT concurrent: a=${short(a.result?.result, 30)} b=${short(b.result?.result, 30)}; peak ${peak.procs} claude processes, ${peak.rssMb.toFixed(0)} MB RSS in total`);
    } else if (check === "interrupt") {
      let release!: () => void;
      const gate = new Promise<void>((r) => { release = r; });
      async function* input() {
        yield { type: "user", message: { role: "user", content: "Count slowly from 1 to 300, one number per line, in words." }, parent_tool_use_id: null, session_id: "" };
        await gate;
        yield { type: "user", message: { role: "user", content: "Reply with just the word DONE." }, parent_tool_use_id: null, session_id: "" };
      }
      const q = query({ prompt: input() as any, options: { model: MODEL, cwd: rig.wt("a"), pathToClaudeCodeExecutable: CLAUDE_BIN, settingSources: [], env: ENV, includePartialMessages: true, maxTurns: 4, ...(BWRAP ? { spawnClaudeCodeProcess: bwrapSpawner(rig, rig.wt("a")) } : {}), ...(SAFE ? { extraArgs: { "safe-mode": null }, strictMcpConfig: true } : {}), canUseTool: async () => ({ behavior: "deny", message: "no" }) } });
      let partials = 0; let results = 0; let interrupted = false;
      const log = fs.createWriteStream(path.join(rig.root, "interrupt.jsonl"));
      for await (const m of q) {
        log.write(JSON.stringify(m) + "\n");
        if (m.type === "stream_event") { partials++; if (partials === 25 && !interrupted) { interrupted = true; console.log("  -> interrupt()"); await q.interrupt(); release(); } }
        if (m.type === "result") { results++; console.log(`  [result #${results}] ${(m as any).subtype} :: ${short((m as any).result, 120)}`); if (results >= 2) break; }
      }
      log.end();
      console.log(`\nRESULT interrupt: partial events=${partials} (>0 means includePartialMessages works), results=${results}`);
    } else {
      console.log("unknown check. one of: login push sandbox hooks resume concurrent interrupt");
    }
  } finally {
    console.log(`\nrig kept at ${rig.root} (messages in <check>.jsonl). Remove with: rm -rf ${rig.root}`);
    if (!flag("keep") && flag("cleanup")) fs.rmSync(rig.root, { recursive: true, force: true });
  }
}

main().catch((e) => { console.error("SPIKE ERROR:", e?.stack ?? e); process.exit(1); });
