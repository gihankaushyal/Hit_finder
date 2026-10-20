import { describe, it, expect } from "vitest";
import fs from "node:fs";
import path from "node:path";
import { normalise, OUTPUT_MAX } from "../server/agents/normalise";

const fixture = (name: string): unknown[] =>
  fs.readFileSync(path.join(__dirname, "fixtures/agents", `${name}.jsonl`), "utf8").trim().split("\n").map((l) => JSON.parse(l));

const run = (name: string) => fixture(name).flatMap((m) => normalise(m));

describe("normalise", () => {
  it("captures the session id from init and nothing else from system and stream messages", () => {
    const items = run("resume");
    expect(items.some((i) => "sessionId" in i && /^[0-9a-f-]{36}$/.test(i.sessionId))).toBe(true);
    for (const type of ["stream_event", "rate_limit_event"]) {
      expect(fixture("push-allow").filter((m) => (m as { type: string }).type === type).flatMap((m) => normalise(m))).toEqual([]);
    }
  });

  it("turns tool calls, their results and the final result into events", () => {
    const bodies = run("push-allow").flatMap((i) => ("body" in i ? [i.body] : []));
    const kinds = bodies.map((b) => b.kind);
    expect(kinds).toContain("tool-use");
    expect(kinds).toContain("tool-result");
    expect(kinds[kinds.length - 1]).toBe("result");
    const use = bodies.find((b) => b.kind === "tool-use");
    expect(use).toMatchObject({ tool: expect.any(String), toolUseId: expect.stringMatching(/^toolu_/), input: expect.any(Object) });
    const last = bodies[bodies.length - 1];
    expect(last).toMatchObject({ kind: "result", ok: true, turns: expect.any(Number) });
  });

  it("pairs each tool result with its tool use", () => {
    const bodies = run("push-allow").flatMap((i) => ("body" in i ? [i.body] : []));
    const uses = new Set(bodies.flatMap((b) => (b.kind === "tool-use" ? [b.toolUseId] : [])));
    for (const b of bodies) if (b.kind === "tool-result") expect(uses.has(b.toolUseId)).toBe(true);
  });

  it("never emits thinking text", () => {
    const text = JSON.stringify(run("push-allow"));
    expect(text).not.toContain("signature");
  });

  it("reports an aborted turn as a failed result, not a crash", () => {
    const bodies = run("interrupt").flatMap((i) => ("body" in i ? [i.body] : []));
    const results = bodies.filter((b) => b.kind === "result");
    expect(results).toHaveLength(2);
    expect(results[0]).toMatchObject({ ok: false });
    expect(results[1]).toMatchObject({ ok: true });
  });

  it("skips the synthetic interruption notice", () => {
    const bodies = run("interrupt").flatMap((i) => ("body" in i ? [i.body] : []));
    expect(JSON.stringify(bodies)).not.toContain("Request interrupted");
  });

  it("flattens array tool output and caps its length", () => {
    const msg = { type: "user", message: { content: [{ type: "tool_result", tool_use_id: "toolu_x", content: [{ type: "text", text: "a" }, { type: "text", text: "b" }], is_error: true }] } };
    expect(normalise(msg)).toEqual([{ body: { kind: "tool-result", toolUseId: "toolu_x", output: "a\nb", isError: true } }]);
    const big = { type: "user", message: { content: [{ type: "tool_result", tool_use_id: "toolu_y", content: "z".repeat(OUTPUT_MAX + 500) }] } };
    const [item] = normalise(big);
    expect("body" in item && item.body.kind === "tool-result" && item.body.output.length).toBeLessThanOrEqual(OUTPUT_MAX + 40);
  });

  it("ignores unknown or malformed messages", () => {
    expect(normalise(null)).toEqual([]);
    expect(normalise({ type: "mystery" })).toEqual([]);
    expect(normalise({ type: "assistant", message: { content: "oops" } })).toEqual([]);
    expect(normalise({ type: "user", message: { content: [{ type: "tool_result" }] } })).toEqual([]);
  });
});
