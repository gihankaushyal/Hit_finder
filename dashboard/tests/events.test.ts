import { describe, it, expect, vi } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { createApp } from "../server/app";
import { loadConfig } from "../server/config";
import { loadOrCreateToken } from "../server/auth";
import { EventBus } from "../server/events";
import { eventRoutes, HEARTBEAT_MS, MAX_QUEUED_EVENTS, QUEUE_STALL_MS } from "../server/routes/events";

function setup(heartbeatMs?: number, limits?: { queueLimit?: number; stallMs?: number }) {
  const stateDir = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "dash-ev-")), "state");
  const token = loadOrCreateToken(stateDir);
  const bus = new EventBus();
  const app = createApp({
    config: { ...loadConfig({}), stateDir, webDist: path.join(stateDir, "no-dist") },
    token,
    bus,
    routes: (api) => eventRoutes(api, { bus, heartbeatMs, ...limits }),
  });
  const open = (ac?: AbortController) =>
    app.request("/api/events", { headers: { Authorization: `Bearer ${token}` }, signal: ac?.signal });
  return { app, bus, open };
}

async function readUntil(reader: ReadableStreamDefaultReader<Uint8Array>, needle: string, max = 20): Promise<string> {
  const dec = new TextDecoder();
  let acc = "";
  for (let i = 0; i < max && !acc.includes(needle); i++) {
    const { value, done } = await reader.read();
    if (done) break;
    acc += dec.decode(value);
  }
  return acc;
}
const waitFor = async (cond: () => boolean) => {
  for (let i = 0; i < 100 && !cond(); i++) await new Promise((r) => setTimeout(r, 10));
};

describe("GET /api/events", () => {
  it("requires the token", async () => {
    const { app, bus } = setup();
    expect((await app.request("/api/events")).status).toBe(401);
    expect(bus.size).toBe(0);
  });

  it("sends hello, forwards bus events, and unsubscribes on cancel", async () => {
    const { bus, open } = setup();
    const before = bus.size;
    const res = await open();
    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toContain("text/event-stream");
    const reader = res.body!.getReader();
    expect(await readUntil(reader, "event: hello")).toContain("event: hello");
    await waitFor(() => bus.size === before + 1);
    bus.emit({ type: "invalidate", resource: "prs" });
    const chunk = await readUntil(reader, '"resource":"prs"');
    expect(chunk).toContain('data: {"type":"invalidate","resource":"prs"}');
    bus.emit({ type: "tests-line", line: "hello world" });
    expect(await readUntil(reader, "hello world")).toContain('"type":"tests-line"');
    await reader.cancel();
    await waitFor(() => bus.size === before);
    expect(bus.size).toBe(before);
  });

  it("unsubscribes when the client cancels the response body and emit does not throw afterwards", async () => {
    const { bus, open } = setup();
    const before = bus.size;
    const ac = new AbortController();
    const res = await open(ac);
    const reader = res.body!.getReader();
    await readUntil(reader, "event: hello");
    await waitFor(() => bus.size === before + 1);
    expect(bus.size).toBe(before + 1);
    ac.abort(); // belt and braces; the cancel below is what the server observes
    reader.cancel().catch(() => {});
    await waitFor(() => bus.size === before);
    expect(bus.size).toBe(before);
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    expect(() => bus.emit({ type: "invalidate", resource: "ci" })).not.toThrow();
    expect(spy).not.toHaveBeenCalled();
    spy.mockRestore();
  });

  it("sends periodic heartbeat comments", async () => {
    expect(HEARTBEAT_MS).toBeLessThanOrEqual(30_000);
    const { open, bus } = setup(20);
    const res = await open();
    const reader = res.body!.getReader();
    expect(await readUntil(reader, ": ping")).toContain(": ping");
    expect(bus.size).toBe(1);
    await reader.cancel();
    await waitFor(() => bus.size === 0);
    expect(bus.size).toBe(0);
  });

  it("drops a client whose queue stays over the limit for the stall time", async () => {
    expect(MAX_QUEUED_EVENTS).toBeGreaterThan(64);
    expect(QUEUE_STALL_MS).toBeGreaterThan(0);
    const { bus, open } = setup(undefined, { queueLimit: 10, stallMs: 40 });
    const before = bus.size;
    const res = await open();
    expect(res.status).toBe(200);
    await waitFor(() => bus.size === before + 1);
    expect(bus.size).toBe(before + 1);
    // Never read from res.body: a fake slow consumer.
    for (let i = 0; i < 2000; i++) bus.emit({ type: "tests-line", line: `l${i}` });
    expect(bus.size).toBe(before + 1); // over the limit, but not for long enough yet
    await new Promise((r) => setTimeout(r, 80));
    bus.emit({ type: "tests-line", line: "late" });
    await waitFor(() => bus.size === before);
    expect(bus.size).toBe(before);
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    expect(() => bus.emit({ type: "invalidate", resource: "ci" })).not.toThrow();
    spy.mockRestore();
    await res.body!.cancel().catch(() => {});
  });

  it("delivers a synchronous burst of 500 events in order to a reading client and keeps it connected", async () => {
    const { bus, open } = setup();
    const res = await open();
    const reader = res.body!.getReader();
    await readUntil(reader, "event: hello");
    await waitFor(() => bus.size === 1);
    const N = 500;
    for (let i = 0; i < N; i++) bus.emit({ type: "tests-line", line: `b${i}` });
    const dec = new TextDecoder();
    let acc = "";
    while (!acc.includes(`"line":"b${N - 1}"`)) {
      const { value, done } = await reader.read();
      if (done) break;
      acc += dec.decode(value);
    }
    const seen = [...acc.matchAll(/"line":"b(\d+)"/g)].map((m) => Number(m[1]));
    expect(seen).toEqual(Array.from({ length: N }, (_, i) => i));
    expect(bus.size).toBe(1);
    await reader.cancel();
  });

  it("keeps a client that reads promptly", async () => {
    const { bus, open } = setup();
    const res = await open();
    const reader = res.body!.getReader();
    await readUntil(reader, "event: hello");
    for (let i = 0; i < 200; i++) {
      bus.emit({ type: "tests-line", line: `l${i}` });
      await readUntil(reader, `l${i}"`);
    }
    expect(bus.size).toBe(1);
    await reader.cancel();
  });
});
