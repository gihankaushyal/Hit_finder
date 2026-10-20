import { describe, it, expect } from "vitest";
import { createFakeDriver, chiefScripts, type Step } from "../server/agents/fakeDriver";
import type { DriverItem, SessionStart } from "../server/agents/driver";
import { loadConfig } from "../server/config";
import { workerOptions } from "../server/agents/policy";
import type { ApprovalOutcome } from "../server/agents/approvals";

const cfg = loadConfig({});
const start = (over: Partial<SessionStart> = {}): SessionStart => ({
  prompt: "do the thing",
  options: workerOptions(cfg, { cwd: "/w", env: {} }),
  claudeBin: "claude",
  canUseTool: async () => ({ decision: "allow" }),
  ...over,
});

async function take(it: AsyncIterator<DriverItem>, until: (i: DriverItem) => boolean): Promise<DriverItem[]> {
  const got: DriverItem[] = [];
  for (;;) {
    const r = await it.next();
    if (r.done) return got;
    got.push(r.value);
    if (until(r.value)) return got;
  }
}
const isResult = (i: DriverItem) => "body" in i && i.body.kind === "result";
const bodies = (items: DriverItem[]) => items.flatMap((i) => ("body" in i ? [i.body] : []));

describe("fake driver", () => {
  it("reports a session id, plays the script and ends the turn with a result", async () => {
    const s = createFakeDriver({ stepMs: 0 }).start(start());
    const it = s.events[Symbol.asyncIterator]();
    const items = await take(it, isResult);
    expect(items[0]).toMatchObject({ sessionId: expect.stringMatching(/^fake-/) });
    expect(bodies(items).map((b) => b.kind)).toEqual(["text", "result"]);
    s.close();
  });

  it("asks before a tool step and records the allowed call and its result", async () => {
    const asked: string[] = [];
    const driver = createFakeDriver({ stepMs: 0, scripts: () => [[{ tool: "Bash", input: { command: "git push origin HEAD" } }]] });
    const s = driver.start(start({ canUseTool: async (tool, input) => (asked.push(`${tool}:${input.command}`), { decision: "allow" }) }));
    const items = await take(s.events[Symbol.asyncIterator](), isResult);
    expect(asked).toEqual(["Bash:git push origin HEAD"]);
    expect(bodies(items).map((b) => b.kind)).toEqual(["tool-use", "tool-result", "result"]);
    s.close();
  });

  it("reports a denied tool as an error result and carries the message", async () => {
    const driver = createFakeDriver({ stepMs: 0, scripts: () => [[{ tool: "Bash", input: { command: "sbatch x" } }]] });
    const s = driver.start(start({ canUseTool: async () => ({ decision: "deny", message: "Not now" }) }));
    const items = bodies(await take(s.events[Symbol.asyncIterator](), isResult));
    expect(items[1]).toMatchObject({ kind: "tool-result", isError: true, output: expect.stringContaining("Not now") });
    s.close();
  });

  it("waits for a decision indefinitely and continues when it arrives", async () => {
    let release!: (o: ApprovalOutcome) => void;
    const driver = createFakeDriver({ stepMs: 0, scripts: () => [[{ tool: "Bash", input: { command: "git push" } }]] });
    const s = driver.start(start({ canUseTool: () => new Promise<ApprovalOutcome>((r) => (release = r)) }));
    const it = s.events[Symbol.asyncIterator]();
    const pending = take(it, isResult);
    await new Promise((r) => setTimeout(r, 30));
    release({ decision: "allow" });
    expect(bodies(await pending).map((b) => b.kind)).toContain("tool-result");
    s.close();
  });

  it("runs another turn for each message sent", async () => {
    const s = createFakeDriver({ stepMs: 0 }).start(start());
    const it = s.events[Symbol.asyncIterator]();
    await take(it, isResult);
    s.send("next thing");
    const second = bodies(await take(it, isResult));
    expect(second[0]).toMatchObject({ kind: "text", text: expect.stringContaining("next thing") });
    expect(second[second.length - 1]).toMatchObject({ kind: "result", ok: true });
    s.close();
  });

  it("interrupting a turn ends it with a failed result and the session keeps working", async () => {
    const slow: Step[] = [{ wait: 5000 }, { say: "never" }];
    const s = createFakeDriver({ stepMs: 0, scripts: (_t, turn) => (turn === 0 ? [slow] : [[{ say: "back" }]]) }).start(start());
    const it = s.events[Symbol.asyncIterator]();
    const first = take(it, isResult);
    await new Promise((r) => setTimeout(r, 20));
    await s.interrupt();
    const items = bodies(await first);
    expect(items[items.length - 1]).toMatchObject({ kind: "result", ok: false });
    s.send("again");
    expect(bodies(await take(it, isResult)).pop()).toMatchObject({ kind: "result", ok: true });
    s.close();
  });

  it("aborts a pending permission request when the session closes", async () => {
    let signal!: AbortSignal;
    const driver = createFakeDriver({ stepMs: 0, scripts: () => [[{ tool: "Bash", input: { command: "git push" } }]] });
    const s = driver.start(start({ canUseTool: (_t, _i, ctx) => ((signal = ctx.signal), new Promise<ApprovalOutcome>(() => {})) }));
    const it = s.events[Symbol.asyncIterator]();
    const p = take(it, () => false);
    await new Promise((r) => setTimeout(r, 20));
    s.close();
    expect(signal.aborted).toBe(true);
    await p;
  });

  it("calls a dashboard tool by name after asking, passing the handler's answer back", async () => {
    const seen: string[] = [];
    const tools = [{ name: "spawn_worker", description: "d", schema: {}, handler: async (a: Record<string, unknown>) => (seen.push(String(a.name)), { text: `started ${a.name}` }) }];
    const asked: string[] = [];
    const driver = createFakeDriver({ stepMs: 0, scripts: () => [[{ mcp: "spawn_worker", input: { name: "w-one" } }]] });
    const s = driver.start(start({ tools, canUseTool: async (t) => (asked.push(t), { decision: "allow" }) }));
    const items = bodies(await take(s.events[Symbol.asyncIterator](), isResult));
    expect(asked).toEqual(["mcp__dashboard__spawn_worker"]);
    expect(seen).toEqual(["w-one"]);
    expect(items.map((b) => b.kind)).toEqual(["tool-use", "tool-result", "result"]);
    expect(items[0]).toMatchObject({ tool: "mcp__dashboard__spawn_worker" });
    expect(items[1]).toMatchObject({ output: "started w-one", isError: false });
    s.close();
  });

  it("does not run the handler when the tool call is denied", async () => {
    let ran = false;
    const tools = [{ name: "stop_worker", description: "d", schema: {}, handler: async () => ((ran = true), { text: "x" }) }];
    const driver = createFakeDriver({ stepMs: 0, scripts: () => [[{ mcp: "stop_worker", input: {} }]] });
    const s = driver.start(start({ tools, canUseTool: async () => ({ decision: "deny", message: "No" }) }));
    const items = bodies(await take(s.events[Symbol.asyncIterator](), isResult));
    expect(ran).toBe(false);
    expect(items[1]).toMatchObject({ isError: true, output: expect.stringContaining("No") });
    s.close();
  });

  it("throws from the event stream when the script fails", async () => {
    const s = createFakeDriver({ stepMs: 0, scripts: () => [[{ fail: "boom" }]] }).start(start());
    await expect(take(s.events[Symbol.asyncIterator](), () => false)).rejects.toThrow("boom");
  });

  it("keeps the session id on resume", async () => {
    const s = createFakeDriver({ stepMs: 0 }).start(start({ resume: "fake-abc" }));
    const items = await take(s.events[Symbol.asyncIterator](), isResult);
    expect(items[0]).toEqual({ sessionId: "fake-abc" });
    s.close();
  });

  it("derives a default script from the task text", async () => {
    const s = createFakeDriver({ stepMs: 0 }).start(start({ prompt: "please push the branch" }));
    const kinds = bodies(await take(s.events[Symbol.asyncIterator](), isResult)).map((b) => b.kind);
    expect(kinds).toContain("tool-use");
    s.close();
  });
});

describe("chiefScripts", () => {
  it("turns 'spawn <name>' into a say step and a spawn_worker tool call", () => {
    const [steps] = chiefScripts("The owner says:\nplease spawn docs-one now", 0);
    expect(steps).toEqual([{ say: "Starting docs-one." }, { mcp: "spawn_worker", input: { name: "docs-one", task: "Scripted task from the chief of staff." } }]);
  });
  it("lists workers on request and otherwise just answers", () => {
    expect(chiefScripts("what workers are running? list", 0)[0]).toEqual([{ mcp: "list_workers", input: {} }]);
    expect(chiefScripts("hello", 0)[0]).toEqual([{ say: "Received: hello" }]);
  });
  it("ignores role text before the owner's message", () => {
    const text = "You are the chief ... spawn_worker ... spawn x\n\nThe owner says:\nhello";
    expect(chiefScripts(text, 0)[0]).toEqual([{ say: "Received: hello" }]);
  });
});
