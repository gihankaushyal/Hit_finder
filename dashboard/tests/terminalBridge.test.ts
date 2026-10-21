import { describe, it, expect } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { fileURLToPath } from "node:url";

const BRIDGE = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../server/terminal/ptyBridge.py");
/** The default interpreter and, where present, the old system one the helper must also run on. */
const PYTHONS = ["python3", ...(fs.existsSync("/usr/bin/python3") ? ["/usr/bin/python3"] : [])];

const frame = (type: "d" | "r", payload: Buffer): Buffer => {
  const head = Buffer.alloc(5);
  head.write(type, 0, "latin1");
  head.writeUInt32BE(payload.length, 1);
  return Buffer.concat([head, payload]);
};
const data = (s: string | Buffer) => frame("d", Buffer.isBuffer(s) ? s : Buffer.from(s));
const resize = (cols: number, rows: number) => {
  const p = Buffer.alloc(4);
  p.writeUInt16BE(cols, 0);
  p.writeUInt16BE(rows, 2);
  return frame("r", p);
};

interface Run {
  child: ChildProcessWithoutNullStreams;
  out(): string;
  exited: Promise<number | null>;
  until(re: RegExp, ms?: number): Promise<string>;
}
function run(python: string, cmd: string[], opts: { cols?: number; rows?: number; env?: Record<string, string> } = {}): Run {
  const args = [BRIDGE, ...(opts.cols ? ["--cols", String(opts.cols)] : []), ...(opts.rows ? ["--rows", String(opts.rows)] : []), "--", ...cmd];
  const child = spawn(python, args, { env: { PATH: process.env.PATH ?? "", ...opts.env } });
  let buf = Buffer.alloc(0);
  child.stdout.on("data", (d: Buffer) => (buf = Buffer.concat([buf, d])));
  const exited = new Promise<number | null>((res) => child.once("exit", (code) => res(code)));
  const out = () => buf.toString("utf8");
  const until = async (re: RegExp, ms = 5000) => {
    const end = Date.now() + ms;
    while (!re.test(out())) {
      if (Date.now() > end) throw new Error(`timed out waiting for ${re}; got ${JSON.stringify(out().slice(-300))}`);
      await new Promise((r) => setTimeout(r, 10));
    }
    return out();
  };
  return { child, out, exited, until };
}

describe.each(PYTHONS)("pty bridge on %s", (python) => {
  it("passes keystrokes to the program and its output back, as a terminal would echo them", async () => {
    const r = run(python, ["cat"]);
    r.child.stdin.write(data("hello\n"));
    await r.until(/hello\r\nhello/);
    r.child.stdin.end();
    await r.exited;
  });

  it("starts the program at the requested size", async () => {
    const r = run(python, ["stty", "size"], { cols: 100, rows: 30 });
    await r.until(/30 100/);
    expect(await r.exited).toBe(0);
  });

  it("resizes the terminal when asked", async () => {
    const r = run(python, ["sh", "-c", "read x; stty size"], { cols: 80, rows: 24 });
    r.child.stdin.write(resize(120, 40));
    r.child.stdin.write(data("\n"));
    await r.until(/40 120/);
    r.child.stdin.end();
    await r.exited;
  });

  it("gives the program a terminal (a tty) and the environment it was started with", async () => {
    const r = run(python, ["sh", "-c", "test -t 0 && echo tty-yes; echo term=$TERM"], { env: { TERM: "xterm-256color" } });
    await r.until(/tty-yes/);
    await r.until(/term=xterm-256color/);
    await r.exited;
  });

  it("ends with the program's own exit status", async () => {
    const r = run(python, ["sh", "-c", "exit 3"]);
    expect(await r.exited).toBe(3);
  });

  it("hangs up and ends the program when its input closes", async () => {
    const pidFile = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "bridge-")), "pid");
    const r = run(python, ["sh", "-c", `echo $$ > ${pidFile}; sleep 60`]);
    await new Promise((res) => setTimeout(res, 400));
    const pid = Number(fs.readFileSync(pidFile, "utf8").trim());
    r.child.stdin.end();
    await Promise.race([r.exited, new Promise((_, rej) => setTimeout(() => rej(new Error("bridge did not exit")), 6000))]);
    await new Promise((res) => setTimeout(res, 200));
    expect(() => process.kill(pid, 0)).toThrow(); // the program is gone
  });

  it("carries a large amount of output without losing or reordering any", async () => {
    const r = run(python, ["seq", "1", "30000"]);
    await r.exited;
    const lines = r.out().split("\r\n").filter(Boolean).map(Number);
    expect(lines).toHaveLength(30000);
    expect(lines[0]).toBe(1);
    expect(lines[29999]).toBe(30000);
    expect(lines.every((n, i) => n === i + 1)).toBe(true);
  });

  it("copes with a frame that arrives in tiny pieces", async () => {
    const r = run(python, ["cat"]);
    for (const byte of data("slow\n")) {
      r.child.stdin.write(Buffer.from([byte]));
      await new Promise((res) => setTimeout(res, 5));
    }
    await r.until(/slow\r\nslow/);
    r.child.stdin.end();
    await r.exited;
  });

  it("passes UTF-8 text through unchanged", async () => {
    const r = run(python, ["cat"]);
    r.child.stdin.write(data("café λ\n"));
    await r.until(/café λ\r\ncafé λ/);
    r.child.stdin.end();
    await r.exited;
  });

  it("ignores a frame type it does not know and keeps working", async () => {
    const r = run(python, ["cat"]);
    r.child.stdin.write(frame("d", Buffer.alloc(0)));
    r.child.stdin.write(Buffer.concat([Buffer.from("z"), Buffer.alloc(4), Buffer.alloc(0)]));
    r.child.stdin.write(data("after\n"));
    await r.until(/after\r\nafter/);
    r.child.stdin.end();
    await r.exited;
  });

  it("refuses to start without a command", async () => {
    const r = run(python, []);
    expect(await r.exited).not.toBe(0);
  });
});
