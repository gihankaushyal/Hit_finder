import type { AgentDriver, DriverItem, Session, SessionStart } from "./driver";

/** One scripted action. A turn is a list of steps; the turn ends with a `result` event. */
export type Step = { say: string } | { tool: string; input: Record<string, unknown> } | { wait: number } | { fail: string };

export interface FakeDriverOptions {
  /** Pause before each step, in milliseconds. Tests use 0; the browser tests use a small value. */
  stepMs?: number;
  /** Script per turn, from the prompt or message that starts it. Turn 0 is the first prompt. */
  scripts?: (text: string, turn: number) => Step[][];
}

/** Default behaviour: say what it is doing, push when the task mentions it, fail when it mentions failing. */
function defaultScripts(text: string, turn: number): Step[][] {
  if (turn > 0) return [[{ say: `Received: ${text}` }]];
  // The supervisor wraps the task in role rules ending "Task:\n<task>"; only the task steers the script.
  const task = text.includes("\nTask:\n") ? (text.split("\nTask:\n").pop() as string) : text;
  const steps: Step[] = [{ say: `Working on: ${task}` }];
  if (/\bpush\b/i.test(task)) steps.push({ tool: "Bash", input: { command: "git push origin HEAD" } });
  if (/\bfail\b/i.test(task)) steps.push({ fail: "Scripted failure" });
  return [steps];
}

type Queued = { item: DriverItem } | { error: Error } | { end: true };

class Channel {
  private queue: Queued[] = [];
  private waiter: (() => void) | null = null;
  push(q: Queued): void {
    this.queue.push(q);
    this.waiter?.();
    this.waiter = null;
  }
  async *iterate(): AsyncGenerator<DriverItem> {
    for (;;) {
      while (this.queue.length === 0) await new Promise<void>((r) => (this.waiter = r));
      const q = this.queue.shift() as Queued;
      if ("end" in q) return;
      if ("error" in q) throw q.error;
      yield q.item;
    }
  }
}

const sleep = (ms: number, signal: AbortSignal) =>
  new Promise<void>((resolve) => {
    if (signal.aborted) return resolve();
    const t = setTimeout(done, ms);
    function done() {
      signal.removeEventListener("abort", done);
      clearTimeout(t);
      resolve();
    }
    signal.addEventListener("abort", done, { once: true });
  });

let counter = 0;

export function createFakeDriver(opts: FakeDriverOptions = {}): AgentDriver {
  const stepMs = opts.stepMs ?? 0;
  const scripts = opts.scripts ?? defaultScripts;
  return {
    start(s: SessionStart): Session {
      const ch = new Channel();
      const life = new AbortController();
      let turnCtl = new AbortController();
      let turn = 0;
      let toolSeq = 0;
      let chain: Promise<void> = Promise.resolve();
      const emit = (body: import("../../shared/types").AgentEventBody) => ch.push({ item: { body } });
      ch.push({ item: { sessionId: s.resume ?? `fake-${++counter}-${Math.random().toString(36).slice(2, 8)}` } });

      const playTurn = async (text: string, index: number): Promise<void> => {
        turnCtl = new AbortController();
        const signal = turnCtl.signal;
        const steps = scripts(text, index)[0] ?? [];
        try {
          for (const step of steps) {
            if (stepMs) await sleep(stepMs, signal);
            if (signal.aborted) break;
            if ("say" in step) emit({ kind: "text", text: step.say });
            else if ("wait" in step) await sleep(step.wait, signal);
            else if ("fail" in step) throw new Error(step.fail);
            else {
              const id = `toolu_fake${++toolSeq}`;
              emit({ kind: "tool-use", tool: step.tool, toolUseId: id, input: step.input });
              const aborted = new Promise<"aborted">((r) => signal.addEventListener("abort", () => r("aborted"), { once: true }));
              const out = await Promise.race([s.canUseTool(step.tool, step.input, { signal }), aborted]);
              if (out === "aborted") break;
              if (out.decision === "allow") emit({ kind: "tool-result", toolUseId: id, output: "ok", isError: false });
              else emit({ kind: "tool-result", toolUseId: id, output: out.message ?? "Denied", isError: true });
            }
          }
          emit({ kind: "result", ok: !signal.aborted, text: signal.aborted ? "Interrupted" : "Done", turns: index + 1, costUsd: 0 });
        } catch (e) {
          ch.push({ error: e instanceof Error ? e : new Error(String(e)) });
          ch.push({ end: true });
          life.abort();
        }
      };

      const enqueue = (text: string) => {
        const index = turn++;
        chain = chain.then(() => (life.signal.aborted ? undefined : playTurn(text, index)));
      };
      enqueue(s.prompt);

      return {
        events: ch.iterate(),
        send: (text) => enqueue(text),
        async interrupt() {
          turnCtl.abort();
        },
        close() {
          if (life.signal.aborted) return;
          life.abort();
          turnCtl.abort();
          ch.push({ end: true });
        },
      };
    },
  };
}
