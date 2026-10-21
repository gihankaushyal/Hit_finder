import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { EventEmitter } from "node:events";
import { PassThrough } from "node:stream";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { createTerminal, terminalEnv, TerminalError, type Terminal, type TerminalDeps } from "../server/terminal/terminal";
import { loadConfig, type Config } from "../server/config";

class FakeChild extends EventEmitter {
  stdin = new PassThrough();
  stdout = new PassThrough();
  pid = 4242;
  written: Buffer[] = [];
  constructor() {
    super();
    this.stdin.on("data", (d: Buffer) => this.written.push(d));
  }
  exit(code = 0) {
    this.stdout.end();
    this.emit("exit", code, null);
  }
}

let cfg: Config;
let children: FakeChild[];
let spawned: { cmd: string; args: string[]; env: Record<string, string | undefined> }[];
let killed: { pid: number; signal: string }[];
let term: Terminal;

function build(over: Partial<Config> = {}, deps: Partial<TerminalDeps> = {}): Terminal {
  cfg = { ...loadConfig({ DASH_REPO_ROOT: "/r/repo" }), ...over };
  return createTerminal({
    cfg,
    env: { PATH: "/bin", HOME: "/h", DASH_TOKEN_FILE: "x", WANDB_API_KEY: "k", GH_TOKEN: "t", ANTHROPIC_API_KEY: "a", DASH_PORT: "1", LANG: "C" },
    spawn: (cmd, args, opts) => {
      const c = new FakeChild();
      children.push(c);
      spawned.push({ cmd, args, env: opts.env });
      return c as never;
    },
    kill: (pid, signal) => void killed.push({ pid, signal }),
    ticketMs: 40,
    killAfterMs: 20,
    ...deps,
  });
}
const frames = (c: FakeChild): { type: string; payload: Buffer }[] => {
  const all = Buffer.concat(c.written);
  const out: { type: string; payload: Buffer }[] = [];
  for (let i = 0; i < all.length; ) {
    const len = all.readUInt32BE(i + 1);
    out.push({ type: all.toString("latin1", i, i + 1), payload: all.subarray(i + 5, i + 5 + len) });
    i += 5 + len;
  }
  return out;
};

beforeEach(() => {
  children = [];
  spawned = [];
  killed = [];
  term = build();
});
afterEach(async () => {
  await term.closeAll(200);
});

describe("open and attach", () => {
  it("opens an unguessable id and starts nothing until it is attached", () => {
    const { id } = term.open(100, 30);
    expect(id).toMatch(/^[0-9a-f-]{36}$/);
    expect(spawned).toHaveLength(0);
    expect(term.count()).toBe(1);
  });

  it("starts the helper with the size, then tmux attached to the session in the repository", () => {
    const { id } = term.open(100, 30);
    term.attach(id);
    const s = spawned[0];
    expect(s.cmd).toBe("python3");
    expect(s.args[0]).toMatch(/ptyBridge\.py$/);
    expect(s.args.slice(1, 5)).toEqual(["--cols", "100", "--rows", "30"]);
    const rest = s.args.slice(s.args.indexOf("--") + 1);
    expect(rest.slice(0, 7)).toEqual(["tmux", "new-session", "-A", "-s", "hitfinder-dash", "-c", "/r/repo"]);
    expect(rest).toContain("status");
  });

  it("uses a private tmux socket when one is configured, and a custom command when given", () => {
    term = build({ terminalTmuxSocket: "test-sock" });
    term.attach(term.open(80, 24).id);
    expect(spawned[0].args).toEqual(expect.arrayContaining(["tmux", "-L", "test-sock"]));
    term = build({ terminalCommand: ["sh", "-c", "cat"] });
    term.attach(term.open(80, 24).id);
    expect(spawned[1].args.slice(spawned[1].args.indexOf("--") + 1)).toEqual(["sh", "-c", "cat"]);
  });

  it("gives the shell a clean environment: no dashboard settings, tokens or keys, and a terminal type", () => {
    term.attach(term.open(80, 24).id);
    const env = spawned[0].env;
    expect(env).toMatchObject({ PATH: "/bin", HOME: "/h", LANG: "C", TERM: "xterm-256color" });
    for (const k of ["DASH_TOKEN_FILE", "DASH_PORT", "WANDB_API_KEY", "GH_TOKEN", "ANTHROPIC_API_KEY"]) expect(env[k]).toBeUndefined();
  });

  it("an id attaches once only", () => {
    const { id } = term.open(80, 24);
    term.attach(id);
    expect(() => term.attach(id)).toThrow(TerminalError);
    expect(spawned).toHaveLength(1);
  });

  it("an id that was never attached expires", async () => {
    const { id } = term.open(80, 24);
    await new Promise((r) => setTimeout(r, 70));
    expect(() => term.attach(id)).toThrow(/not found/i);
    expect(term.count()).toBe(0);
  });

  it("refuses unknown ids", () => {
    expect(() => term.attach("nope")).toThrow(/not found/i);
    expect(() => term.input("nope", "x")).toThrow(/not found/i);
    expect(() => term.resize("nope", 80, 24)).toThrow(/not found/i);
  });

  it("enforces the stream limit, counting ids not yet attached, and frees a slot on close", async () => {
    const a = term.open(80, 24);
    term.open(80, 24);
    expect(() => term.open(80, 24)).toThrow(expect.objectContaining({ code: "limit" }));
    await term.close(a.id);
    expect(() => term.open(80, 24)).not.toThrow();
  });

  it("refuses everything when the terminal is switched off", () => {
    term = build({ terminalEnabled: false });
    expect(term.enabled).toBe(false);
    expect(() => term.open(80, 24)).toThrow(expect.objectContaining({ code: "disabled" }));
  });

  it("rejects absurd sizes", () => {
    for (const [c, r] of [[0, 24], [80, 0], [1000, 24], [80, 500], [1.5, 24], [NaN, 24]]) {
      expect(() => term.open(c, r)).toThrow(expect.objectContaining({ code: "invalid" }));
    }
  });
});

describe("input and resize", () => {
  it("sends keystrokes and resizes to the helper as frames, in order", () => {
    const { id } = term.open(80, 24);
    term.attach(id);
    term.input(id, "ls\r");
    term.resize(id, 120, 40);
    term.input(id, "é");
    const f = frames(children[0]);
    expect(f.map((x) => x.type)).toEqual(["d", "r", "d"]);
    expect(f[0].payload.toString()).toBe("ls\r");
    expect(f[1].payload.readUInt16BE(0)).toBe(120);
    expect(f[1].payload.readUInt16BE(2)).toBe(40);
    expect(f[2].payload.toString("utf8")).toBe("é");
  });

  it("refuses input before the stream is attached, and oversized or empty input", () => {
    const { id } = term.open(80, 24);
    expect(() => term.input(id, "x")).toThrow(expect.objectContaining({ code: "state" }));
    term.attach(id);
    expect(() => term.input(id, "x".repeat(70_000))).toThrow(expect.objectContaining({ code: "invalid" }));
    expect(() => term.input(id, "")).toThrow(expect.objectContaining({ code: "invalid" }));
    expect(() => term.resize(id, 0, 24)).toThrow(expect.objectContaining({ code: "invalid" }));
  });
});

describe("lifecycle", () => {
  it("the output stream is the helper's output, and ends with it", async () => {
    const { id } = term.open(80, 24);
    const { output } = term.attach(id);
    const got: Buffer[] = [];
    output.on("data", (d: Buffer) => got.push(d));
    children[0].stdout.write("hello");
    await new Promise((r) => setTimeout(r, 10));
    expect(Buffer.concat(got).toString()).toBe("hello");
    children[0].exit(0);
    await new Promise((r) => setTimeout(r, 10));
    expect(term.count()).toBe(0);
    expect(() => term.input(id, "x")).toThrow(/not found/i);
  });

  it("close ends the helper's input, hangs it up, and escalates to a kill if it lingers", async () => {
    const { id } = term.open(80, 24);
    term.attach(id);
    await term.close(id);
    expect(children[0].stdin.writableEnded).toBe(true);
    expect(killed.map((k) => k.signal)).toEqual(["SIGTERM", "SIGKILL"]);
    expect(killed[0].pid).toBe(-4242); // the whole process group
    expect(term.count()).toBe(0);
  });

  it("close of a helper that exits promptly does not need the kill", async () => {
    const { id } = term.open(80, 24);
    term.attach(id);
    const closing = term.close(id);
    children[0].exit(0);
    await closing;
    await new Promise((r) => setTimeout(r, 50));
    expect(killed.map((k) => k.signal)).toEqual(["SIGTERM"]);
  });

  it("closeAll closes every stream within the time limit", async () => {
    term.attach(term.open(80, 24).id);
    term.attach(term.open(80, 24).id);
    await term.closeAll(500);
    expect(term.count()).toBe(0);
    expect(killed.filter((k) => k.signal === "SIGTERM")).toHaveLength(2);
  });

  it("close of an unknown id is quiet", async () => {
    await expect(term.close("nope")).resolves.toBeUndefined();
  });
});

describe("terminalEnv", () => {
  it("passes a short list of ordinary variables and nothing else, so no credential name has to be guessed", () => {
    const env = terminalEnv({
      PATH: "/bin", HOME: "/h", USER: "u", LOGNAME: "u", SHELL: "/bin/bash", LANG: "C", LC_ALL: "C", TZ: "UTC", TMPDIR: "/t",
      A: "1", DASH_X: "2", MY_TOKEN: "3", AWS_ACCESS_KEY_ID: "4", DATABASE_URL: "postgres://u:pw@h/db", SSH_AUTH_SOCK: "/s", WANDB_API_KEY: "6", SLURM_JOB_ID: "7", GIT_ASKPASS: "x",
    });
    expect(Object.keys(env).sort()).toEqual(["COLORTERM", "HOME", "LANG", "LC_ALL", "LOGNAME", "PATH", "SHELL", "TERM", "TMPDIR", "TZ", "USER"]);
  });
});

describe("a helper that cannot start", () => {
  it("is dropped without taking the server down, and frees its slot at once", async () => {
    const { id } = term.open(80, 24);
    const { output } = term.attach(id);
    let ended = false;
    output.on("end", () => (ended = true)).on("close", () => (ended = true));
    children[0].emit("error", Object.assign(new Error("spawn python3 ENOENT"), { code: "ENOENT" }));
    await new Promise((r) => setTimeout(r, 20));
    expect(term.count()).toBe(0);
    expect(ended).toBe(true);
    expect(() => term.input(id, "x")).toThrow(/not found/i);
  });

  it("with a real missing interpreter nothing is thrown and the stream just ends", async () => {
    const real = createTerminal({ cfg: { ...loadConfig({ DASH_REPO_ROOT: "/tmp" }), terminalPython: "/nonexistent/python-for-test" }, env: { PATH: "/bin" } });
    const { id } = real.open(80, 24);
    const { output } = real.attach(id);
    const ended = new Promise<void>((res) => { output.on("close", () => res()); output.on("end", () => res()); output.resume(); });
    await Promise.race([ended, new Promise((_, rej) => setTimeout(() => rej(new Error("stream never ended")), 3000))]);
    expect(real.count()).toBe(0);
    await expect(real.close(id)).resolves.toBeUndefined();
  });
});

const haveTmux = spawnSync("tmux", ["-V"]).status === 0 && spawnSync("python3", ["--version"]).status === 0;
describe.skipIf(!haveTmux)("with a real tmux on a private socket", () => {
  const socket = `dash-test-${process.pid}`;
  afterEach(() => void spawnSync("tmux", ["-L", socket, "kill-server"]));

  /** Collects a stream's text once, so several waits can look at the same output. */
  function collect(output: NodeJS.ReadableStream) {
    let text = "";
    output.on("data", (d: Buffer) => (text += d.toString("utf8")));
    return async (re: RegExp, ms = 12_000): Promise<string> => {
      const end = Date.now() + ms;
      while (!re.test(text)) {
        if (Date.now() > end) throw new Error(`timed out waiting for ${re}; got ${JSON.stringify(text.slice(-300))}`);
        await new Promise((r) => setTimeout(r, 25));
      }
      return text;
    };
  }

  it("runs a command, and a second attach finds the same session with its history", { timeout: 40_000 }, async () => {
    const real = createTerminal({ cfg: { ...loadConfig({ DASH_REPO_ROOT: path.resolve("/tmp") }), terminalTmuxSocket: socket, terminalSession: "dash-test" }, env: { ...process.env } });
    const first = real.open(100, 30);
    const seen = collect(real.attach(first.id).output);
    await seen(/\$ $/); // wait for the prompt, as a person would before typing
    real.input(first.id, "echo answer-$((40+2))\r");
    await seen(/\r\nanswer-42/);
    await real.close(first.id);
    const second = real.open(100, 30);
    const again = collect(real.attach(second.id).output);
    await again(/answer-42/);
    await real.closeAll(2000);
    expect(real.count()).toBe(0);
  });
});
