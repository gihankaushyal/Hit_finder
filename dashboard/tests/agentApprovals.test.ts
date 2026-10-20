import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { createApprovals, summarise, type Approvals } from "../server/agents/approvals";
import { EventBus } from "../server/events";
import type { DashEvent } from "../shared/types";

let dir: string;
let audit: string;
let bus: EventBus;
let events: DashEvent[];
let ap: Approvals;
const agent = { id: "a1", name: "fix-lcn" };

beforeEach(() => {
  dir = fs.mkdtempSync(path.join(os.tmpdir(), "dash-ap-"));
  audit = path.join(dir, "audit.jsonl");
  bus = new EventBus();
  events = [];
  bus.subscribe((e) => events.push(e));
  ap = createApprovals({ bus, auditPath: audit });
});
afterEach(() => fs.rmSync(dir, { recursive: true, force: true }));

describe("request and decide", () => {
  it("holds a request until it is decided, then resolves it once", async () => {
    let settled: unknown;
    const p = ap.request(agent, "Bash", { command: "git push origin HEAD" }).then((d) => (settled = d));
    await Promise.resolve();
    expect(settled).toBeUndefined();
    const [a] = ap.pending();
    expect(a).toMatchObject({ agentId: "a1", agentName: "fix-lcn", tool: "Bash", input: { command: "git push origin HEAD" } });
    ap.decide(a.id, "allow");
    expect(await p).toEqual({ decision: "allow" });
    expect(ap.pending()).toEqual([]);
  });

  it("carries a deny message back to the agent", async () => {
    const p = ap.request(agent, "Bash", { command: "sbatch x.sh" });
    ap.decide(ap.pending()[0].id, "deny", "Not now");
    expect(await p).toEqual({ decision: "deny", message: "Not now" });
  });

  it("rejects an unknown id and a repeated decision", async () => {
    const p = ap.request(agent, "Bash", { command: "git push" });
    const id = ap.pending()[0].id;
    expect(() => ap.decide("nope", "allow")).toThrow(/not found/i);
    ap.decide(id, "allow");
    await p;
    expect(() => ap.decide(id, "deny")).toThrow(/not found/i);
  });

  it("uses unguessable ids", async () => {
    ap.request(agent, "Bash", { command: "a" });
    ap.request(agent, "Bash", { command: "b" });
    const ids = ap.pending().map((a) => a.id);
    expect(new Set(ids).size).toBe(2);
    for (const id of ids) expect(id).toMatch(/^[0-9a-f-]{36}$/);
  });

  it("lists pending requests oldest first and counts them", async () => {
    vi.useFakeTimers();
    try {
      ap.request(agent, "Bash", { command: "first" });
      vi.advanceTimersByTime(1000);
      ap.request({ id: "a2", name: "other" }, "Bash", { command: "second" });
      expect(ap.pending().map((a) => a.input.command)).toEqual(["first", "second"]);
      expect(ap.count()).toBe(2);
      expect(ap.countFor("a2")).toBe(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it("never decides on its own: no timeout", async () => {
    vi.useFakeTimers();
    try {
      ap.request(agent, "Bash", { command: "git push" });
      vi.advanceTimersByTime(7 * 24 * 3600 * 1000);
      expect(ap.count()).toBe(1);
    } finally {
      vi.useRealTimers();
    }
  });
});

describe("onCreated", () => {
  it("reports the new approval before anything is decided", async () => {
    let seen: string | undefined;
    const p = ap.request(agent, "Bash", { command: "x" }, { onCreated: (a) => (seen = a.id) });
    expect(seen).toBe(ap.pending()[0].id);
    ap.decide(seen as string, "allow");
    await p;
  });
});

describe("cancellation", () => {
  it("cancels every pending request of one agent as a denial", async () => {
    const a = ap.request(agent, "Bash", { command: "x" });
    const b = ap.request({ id: "a2", name: "o" }, "Bash", { command: "y" });
    ap.cancelAgent("a1");
    expect(await a).toMatchObject({ decision: "deny", cancelled: true });
    expect(ap.count()).toBe(1);
    ap.cancelAgent("a2");
    expect(await b).toMatchObject({ decision: "deny", cancelled: true });
  });

  it("cancels one request when its abort signal fires", async () => {
    const ac = new AbortController();
    const p = ap.request(agent, "Bash", { command: "x" }, { signal: ac.signal });
    ac.abort();
    expect(await p).toMatchObject({ decision: "deny", cancelled: true });
    expect(ap.count()).toBe(0);
  });

  it("denies at once when the signal is already aborted", async () => {
    const ac = new AbortController();
    ac.abort();
    expect(await ap.request(agent, "Bash", { command: "x" }, { signal: ac.signal })).toMatchObject({ decision: "deny", cancelled: true });
    expect(ap.count()).toBe(0);
  });

  it("deciding a cancelled request is an error", async () => {
    const p = ap.request(agent, "Bash", { command: "x" });
    const id = ap.pending()[0].id;
    ap.cancelAgent("a1");
    await p;
    expect(() => ap.decide(id, "allow")).toThrow(/not found/i);
  });
});

describe("notifications and audit", () => {
  it("emits an approvals invalidate on request, decision and cancel", async () => {
    const p = ap.request(agent, "Bash", { command: "x" });
    ap.decide(ap.pending()[0].id, "allow");
    await p;
    const q = ap.request(agent, "Bash", { command: "y" });
    ap.cancelAgent("a1");
    await q;
    expect(events.filter((e) => e.type === "invalidate" && e.resource === "approvals")).toHaveLength(4);
  });

  it("appends a record for every decision and cancellation", async () => {
    const p = ap.request(agent, "Bash", { command: "git push origin HEAD" });
    ap.decide(ap.pending()[0].id, "deny");
    await p;
    const q = ap.request(agent, "Edit", { file_path: "/w/x.ts" });
    ap.cancelAgent("a1");
    await q;
    const lines = fs.readFileSync(audit, "utf8").trim().split("\n").map((l) => JSON.parse(l));
    expect(lines).toHaveLength(2);
    expect(lines[0]).toMatchObject({ agentId: "a1", agentName: "fix-lcn", tool: "Bash", summary: "git push origin HEAD", decision: "deny" });
    expect(lines[1]).toMatchObject({ tool: "Edit", decision: "cancelled" });
    expect(new Date(lines[0].at).toISOString()).toBe(lines[0].at);
  });

  it("still works if the audit file cannot be written", async () => {
    const bad = createApprovals({ bus, auditPath: path.join(dir, "no", "such", "dir", "a.jsonl") });
    const p = bad.request(agent, "Bash", { command: "x" });
    bad.decide(bad.pending()[0].id, "allow");
    expect(await p).toEqual({ decision: "allow" });
  });
});

describe("summarise", () => {
  it("shows the first line of a Bash command, shortened", () => {
    expect(summarise("Bash", { command: "git push\nsecond" })).toBe("git push");
    expect(summarise("Bash", { command: "x".repeat(500) }).length).toBeLessThanOrEqual(200);
  });
  it("names the file for edits and writes", () => {
    expect(summarise("Edit", { file_path: "/w/a.ts" })).toBe("Edit /w/a.ts");
    expect(summarise("Write", { file_path: "/w/b.ts", content: "x" })).toBe("Write /w/b.ts");
  });
  it("falls back to the tool name and compact input", () => {
    expect(summarise("mcp__x__go", { a: 1 })).toBe('mcp__x__go {"a":1}');
  });
});
