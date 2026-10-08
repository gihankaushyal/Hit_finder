import { describe, it, expect, afterEach, vi } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { Hono } from "hono";
import { createTestRunner, type SpawnFn, type TestRunner } from "../server/lib/testRunner";
import { testRoutes } from "../server/routes/tests";
import { EventBus } from "../server/events";
import type { Config } from "../server/config";
import type { Runner } from "../server/runner";
import type { DashEvent } from "../shared/types";

const dirs: string[] = [];
afterEach(() => {
  while (dirs.length) fs.rmSync(dirs.pop()!, { recursive: true, force: true });
});

const JUNIT_FAIL = `<testsuite tests="3" failures="1" errors="0" skipped="0" time="1.5">
<testcase classname="tests.t" name="ok"/><testcase classname="tests.t" name="ok2"/>
<testcase classname="tests.t" name="bad"><failure message="assert 0">x</failure></testcase></testsuite>`;
const JUNIT_PASS = `<testsuite tests="1" failures="0" errors="0" skipped="0" time="0.1"><testcase classname="t" name="ok"/></testsuite>`;

function setup(stateDir?: string) {
  const dir = stateDir ?? fs.mkdtempSync(path.join(os.tmpdir(), "dash-tr-"));
  if (!stateDir) dirs.push(dir);
  const config = { repoRoot: "/repo", stateDir: dir, pythonBin: "py" } as Config;
  const spawnCalls: { cmd: string; args: string[]; cwd: string }[] = [];
  let emit: (l: string) => void = () => {};
  let finish: (code: number) => void = () => {};
  let killed = 0;
  const spawn: SpawnFn = (cmd, args, opts) => {
    spawnCalls.push({ cmd, args, cwd: opts.cwd });
    return {
      onLine: (fn) => { emit = fn; },
      done: new Promise<number>((res) => { finish = res; }),
      kill: () => { killed++; },
    };
  };
  const runnerCalls: { cmd: string; args: string[]; cwd?: string }[] = [];
  const runner: Runner = async (cmd, args, opts) => {
    runnerCalls.push({ cmd, args, cwd: opts?.cwd });
    return { stdout: "abc1234\n", stderr: "", code: 0 };
  };
  const bus = new EventBus();
  const events: DashEvent[] = [];
  bus.subscribe((e) => events.push(e));
  const tr = createTestRunner({ config, spawn, runner, bus });
  return {
    tr, dir, bus, events, spawnCalls, runnerCalls, config, spawn, runner,
    emit: (l: string) => emit(l), finish: (c: number) => finish(c), killed: () => killed,
  };
}
const settle = async (tr: TestRunner) => {
  for (let i = 0; i < 100 && tr.state().status === "running"; i++) await new Promise((r) => setTimeout(r, 5));
};
const waitSpawned = async (s: ReturnType<typeof setup>) => {
  for (let i = 0; i < 100 && s.spawnCalls.length === 0; i++) await new Promise((r) => setTimeout(r, 5));
};

describe("test runner", () => {
  it("spawns pytest with exact args, records commit, streams lines, and parses a passing report", async () => {
    const s = setup();
    expect(s.tr.state().status).toBe("idle");
    expect(s.tr.start()).toBe(true);
    expect(s.tr.state().status).toBe("running");
    await waitSpawned(s);
    expect(s.spawnCalls[0]).toEqual({
      cmd: "py",
      args: ["-m", "pytest", "tests/", "-q", `--junitxml=${path.join(s.dir, "junit.xml")}`],
      cwd: "/repo",
    });
    expect(s.runnerCalls[0]).toMatchObject({ cmd: "git", args: ["rev-parse", "--short", "HEAD"], cwd: "/repo" });
    s.emit("line one");
    s.emit("line two");
    expect(s.tr.state().tail).toEqual(["line one", "line two"]);
    expect(s.events).toContainEqual({ type: "tests-line", line: "line one" });
    fs.writeFileSync(path.join(s.dir, "junit.xml"), JUNIT_PASS);
    s.finish(0);
    await settle(s.tr);
    const st = s.tr.state();
    expect(st.status).toBe("done");
    expect(st.commit).toBe("abc1234");
    expect(st.summary).toMatchObject({ tests: 1, passed: 1, failed: 0 });
    expect(st.finishedAt).not.toBeNull();
    const inv = s.events.filter((e) => e.type === "invalidate" && e.resource === "tests");
    expect(inv).toHaveLength(2);
  });

  it("start() returns false while a run is active", async () => {
    const s = setup();
    expect(s.tr.start()).toBe(true);
    expect(s.tr.start()).toBe(false);
    await waitSpawned(s);
    expect(s.spawnCalls).toHaveLength(1);
    s.finish(1);
    await settle(s.tr);
    expect(s.tr.start()).toBe(true);
  });

  it("reports failing test names when pytest exits 1 with a report", async () => {
    const s = setup();
    s.tr.start();
    await waitSpawned(s);
    fs.writeFileSync(path.join(s.dir, "junit.xml"), JUNIT_FAIL);
    s.finish(1);
    await settle(s.tr);
    const st = s.tr.state();
    expect(st.status).toBe("done");
    expect(st.summary?.failed).toBe(1);
    expect(st.summary?.failures).toEqual([{ name: "tests.t::bad", message: "assert 0" }]);
  });

  it("deletes a stale junit.xml before running, so a crash is not mistaken for success", async () => {
    const s = setup();
    fs.writeFileSync(path.join(s.dir, "junit.xml"), JUNIT_PASS);
    s.tr.start();
    await waitSpawned(s);
    expect(fs.existsSync(path.join(s.dir, "junit.xml"))).toBe(false);
    s.finish(2);
    await settle(s.tr);
    expect(s.tr.state().status).toBe("error");
  });

  it("error state when pytest exits non-zero and writes no report", async () => {
    const s = setup();
    s.tr.start();
    await waitSpawned(s);
    s.finish(2);
    await settle(s.tr);
    const st = s.tr.state();
    expect(st.status).toBe("error");
    expect(st.error).toBe("pytest exited with code 2 and wrote no report");
    expect(st.summary).toBeNull();
  });

  it("error state (and not stuck running) when the report is malformed", async () => {
    const s = setup();
    s.tr.start();
    await waitSpawned(s);
    fs.writeFileSync(path.join(s.dir, "junit.xml"), "<testsuite><testcase");
    s.finish(1);
    await settle(s.tr);
    const st = s.tr.state();
    expect(st.status).toBe("error");
    expect(st.error).toMatch(/report/i);
    expect(s.tr.start()).toBe(true);
  });

  it("keeps only the last 200 lines", async () => {
    const s = setup();
    s.tr.start();
    await waitSpawned(s);
    for (let i = 0; i < 250; i++) s.emit(`l${i}`);
    const tail = s.tr.state().tail;
    expect(tail).toHaveLength(200);
    expect(tail[0]).toBe("l50");
    expect(tail[199]).toBe("l249");
    s.finish(1);
    await settle(s.tr);
  });

  it("persists the final state and a new runner on the same stateDir reports it", async () => {
    const s = setup();
    s.tr.start();
    await waitSpawned(s);
    fs.writeFileSync(path.join(s.dir, "junit.xml"), JUNIT_FAIL);
    s.finish(1);
    await settle(s.tr);
    const s2 = setup(s.dir);
    expect(s2.tr.state()).toEqual(s.tr.state());
    expect(s2.tr.state().status).toBe("done");
  });

  it("ignores a corrupt persisted state file", () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "dash-tr-"));
    dirs.push(dir);
    fs.writeFileSync(path.join(dir, "tests-last.json"), "{nope");
    expect(setup(dir).tr.state().status).toBe("idle");
  });

  it("stop() kills the child process of an active run", async () => {
    const s = setup();
    s.tr.start();
    await waitSpawned(s);
    s.tr.stop();
    expect(s.killed()).toBe(1);
    s.finish(137);
    await settle(s.tr);
  });

  it("works with a missing commit (git failure) and a rejecting spawn", async () => {
    const s = setup();
    const failing: Runner = async () => ({ stdout: "", stderr: "x", code: 128 });
    const bad: SpawnFn = () => {
      throw new Error("boom");
    };
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    const tr = createTestRunner({ config: s.config, spawn: bad, runner: failing, bus: s.bus });
    tr.start();
    await settle(tr);
    expect(tr.state().status).toBe("error");
    expect(tr.state().commit).toBeNull();
    expect(tr.state().error).toBe("could not start pytest");
    spy.mockRestore();
  });
});

describe("test runner failure paths", () => {
  it("ends in an error state, not running, when done rejects", async () => {
    const s = setup();
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    const rejecting: SpawnFn = () => ({ onLine: () => {}, done: Promise.reject(new Error("pipe broke")), kill: () => {} });
    const tr = createTestRunner({ config: s.config, spawn: rejecting, runner: s.runner, bus: s.bus });
    expect(tr.start()).toBe(true);
    await settle(tr);
    expect(tr.state().status).toBe("error");
    expect(tr.state().error).toBe("could not start pytest");
    expect(tr.start()).toBe(true);
    spy.mockRestore();
  });
  it("still runs and ends non-running when the git runner rejects", async () => {
    const s = setup();
    const throwing: Runner = async () => {
      throw new Error("no git");
    };
    const tr = createTestRunner({ config: s.config, spawn: s.spawn, runner: throwing, bus: s.bus });
    tr.start();
    await waitSpawned(s);
    s.finish(2);
    await settle(tr);
    expect(tr.state().status).toBe("error");
    expect(tr.state().commit).toBeNull();
  });
  it("truncates over-long lines in the tail and on the bus", async () => {
    const s = setup();
    s.tr.start();
    await waitSpawned(s);
    s.emit("x".repeat(5000));
    expect(s.tr.state().tail[0]).toHaveLength(2000);
    const ev = s.events.find((e) => e.type === "tests-line");
    expect(ev?.type === "tests-line" && ev.line.length).toBe(2000);
    s.finish(1);
    await settle(s.tr);
    expect(JSON.parse(fs.readFileSync(path.join(s.dir, "tests-last.json"), "utf8")).tail[0]).toHaveLength(2000);
  });
});

describe("test routes", () => {
  function routes() {
    const s = setup();
    const api = new Hono();
    testRoutes(api, { tests: s.tr });
    return { api, s };
  }
  it("GET /tests returns state; POST /tests/run returns 202 then 409", async () => {
    const { api, s } = routes();
    expect(((await (await api.request("/tests")).json()) as { status: string }).status).toBe("idle");
    const r1 = await api.request("/tests/run", { method: "POST" });
    expect(r1.status).toBe(202);
    expect(await r1.json()).toEqual({ started: true });
    const r2 = await api.request("/tests/run", { method: "POST" });
    expect(r2.status).toBe(409);
    expect(await r2.json()).toEqual({ error: "A test run is already in progress" });
    expect(s.spawnCalls.length).toBeLessThanOrEqual(1);
    await waitSpawned(s);
    s.finish(1);
    await settle(s.tr);
  });
});

describe("nodeSpawn", () => {
  it("streams stdout and stderr lines and reports the exit code", async () => {
    const { nodeSpawn } = await import("../server/lib/testRunner");
    const h = nodeSpawn(process.execPath, ["-e", "console.log('a'); console.error('b'); process.exit(3)"], { cwd: process.cwd() });
    const lines: string[] = [];
    h.onLine((l) => lines.push(l));
    expect(await h.done).toBe(3);
    expect(lines.sort()).toEqual(["a", "b"]);
  });
  it("reports 127 for a missing binary and kill() ends a long-running child", async () => {
    const { nodeSpawn } = await import("../server/lib/testRunner");
    expect(await nodeSpawn("/nonexistent/bin", [], { cwd: process.cwd() }).done).toBe(127);
    const h = nodeSpawn(process.execPath, ["-e", "setTimeout(()=>{},60000)"], { cwd: process.cwd() });
    h.kill();
    expect(await h.done).not.toBe(0);
  });
});

function alive(pid: number): boolean {
  try {
    process.kill(pid, 0);
  } catch {
    return false;
  }
  try {
    // A zombie that nobody has reaped yet counts as dead.
    return !/^\d+ \(.*\) Z/.test(fs.readFileSync(`/proc/${pid}/stat`, "utf8"));
  } catch {
    return true;
  }
}
const until = async (cond: () => boolean, ms = 4000) => {
  for (let i = 0; i < ms / 20 && !cond(); i++) await new Promise((r) => setTimeout(r, 20));
};

describe("nodeSpawn process control", () => {
  it("kill() terminates the whole process group, including grandchildren", async () => {
    const { nodeSpawn } = await import("../server/lib/testRunner");
    const script = [
      "const c = require('child_process').spawn(process.execPath, ['-e', 'setTimeout(()=>{},60000)'], { stdio: 'ignore' });",
      "console.log('PIDS ' + process.pid + ' ' + c.pid);",
      "setTimeout(()=>{},60000);",
    ].join("");
    const h = nodeSpawn(process.execPath, ["-e", script], { cwd: process.cwd() });
    let pids: number[] = [];
    h.onLine((l) => {
      const m = /^PIDS (\d+) (\d+)$/.exec(l);
      if (m) pids = [Number(m[1]), Number(m[2])];
    });
    await until(() => pids.length === 2);
    expect(pids).toHaveLength(2);
    expect(pids.every(alive)).toBe(true);
    h.kill();
    await h.done;
    await until(() => pids.every((p) => !alive(p)));
    expect(pids.map(alive)).toEqual([false, false]);
  });

  it("done resolves shortly after exit even if a leftover process holds the output pipes", async () => {
    const { nodeSpawn } = await import("../server/lib/testRunner");
    const script =
      "require('child_process').spawn(process.execPath, ['-e', 'setTimeout(()=>{},4000)'], { detached: true, stdio: ['ignore', 1, 1] }).unref();";
    const t0 = Date.now();
    const h = nodeSpawn(process.execPath, ["-e", script], { cwd: process.cwd() });
    expect(await h.done).toBe(0);
    expect(Date.now() - t0).toBeLessThan(2500);
  });
});
