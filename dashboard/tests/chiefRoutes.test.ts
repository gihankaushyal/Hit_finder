import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { makeAgentRig, until, AUTH, AGENT_TOKEN, type AgentRig } from "./helpers/agentRig";
import type { AgentEventsResponse, ApprovalsResponse, ChiefSummary } from "../shared/types";

let rig: AgentRig;
beforeEach(async () => {
  rig = await makeAgentRig({}, {
    stepMs: 0,
    scripts: (t) => (t.includes("start a worker") ? [[{ mcp: "spawn_worker", input: { name: "chief-made", task: "scripted task" } }]] : [[{ say: "hello" }]]),
  });
});
afterEach(() => rig.cleanup());

const post = (url: string, body?: unknown) => rig.app.request(url, { method: "POST", headers: AUTH, body: body === undefined ? undefined : JSON.stringify(body) });
const get = (url: string) => rig.app.request(url, { headers: AUTH });

describe("chief routes", () => {
  it("reports an idle chief before any message", async () => {
    const res = await get("/api/chief");
    expect(res.status).toBe(200);
    expect((await res.json()) as ChiefSummary).toMatchObject({ state: "idle", model: "opus", sessionId: null, turns: 0, pendingApprovals: 0 });
  });

  it("takes a message, answers, and serves the log by cursor", async () => {
    expect((await post("/api/chief/message", { text: "hi chief" })).status).toBe(200);
    await until(() => rig.chief.summary().turns === 1 || undefined);
    const log = (await (await get("/api/chief/events")).json()) as AgentEventsResponse;
    expect(log.events[0]).toMatchObject({ kind: "user", text: "hi chief" });
    expect(log.events.map((e) => e.kind)).toContain("text");
    const rest = (await (await get(`/api/chief/events?after=${log.events[0].seq}&limit=1`)).json()) as AgentEventsResponse;
    expect(rest.events).toHaveLength(1);
    expect(rest.events[0].seq).toBe(log.events[1].seq);
  });

  it("validates messages and cursors", async () => {
    expect((await post("/api/chief/message", { text: "" })).status).toBe(400);
    expect((await post("/api/chief/message", { text: 4 })).status).toBe(400);
    expect((await post("/api/chief/message", [1])).status).toBe(400);
    expect((await get("/api/chief/events?after=-1")).status).toBe(400);
    expect((await get("/api/chief/events?limit=9999")).status).toBe(400);
  });

  it("a worker spawned by the chief waits on a card, and runs only after approval", async () => {
    await post("/api/chief/message", { text: "please start a worker" });
    await until(() => rig.chief.summary().state === "waiting-approval" || undefined);
    const cards = ((await (await get("/api/approvals")).json()) as ApprovalsResponse).approvals;
    expect(cards).toHaveLength(1);
    expect(cards[0]).toMatchObject({ agentName: "Chief of staff", tool: "mcp__dashboard__spawn_worker", input: { name: "chief-made" } });
    expect(rig.sup.list()).toEqual([]);
    expect((await post(`/api/approvals/${cards[0].id}/decision`, { decision: "allow" })).status).toBe(200);
    await until(() => rig.sup.get("chief-made") || undefined);
    await until(() => rig.chief.summary().state === "idle" || undefined);
  });

  it("stop and reset work and report plainly", async () => {
    await post("/api/chief/message", { text: "please start a worker" });
    await until(() => rig.chief.summary().state === "waiting-approval" || undefined);
    expect((await post("/api/chief/stop")).status).toBe(200);
    expect(rig.chief.summary().state).toBe("stopped");
    expect(rig.approvals.count()).toBe(0);
    expect((await post("/api/chief/reset")).status).toBe(200);
    expect(rig.chief.summary()).toMatchObject({ state: "idle", sessionId: null });
  });

  it.each([
    ["GET", "/api/chief"],
    ["POST", "/api/chief/message"],
    ["GET", "/api/chief/events"],
    ["POST", "/api/chief/stop"],
    ["POST", "/api/chief/reset"],
  ])("%s %s without a token is 401", async (method, url) => {
    expect((await rig.app.request(url, { method })).status).toBe(401);
    expect(rig.chief.summary().state).toBe("idle");
  });

  it("refuses a cross-origin message even with the cookie", async () => {
    const res = await rig.app.request("http://127.0.0.1:4317/api/chief/message", {
      method: "POST",
      headers: { Cookie: `dash_token=${AGENT_TOKEN}`, "Content-Type": "application/json", Origin: "https://evil.example" },
      body: JSON.stringify({ text: "hi" }),
    });
    expect(res.status).toBe(403);
    expect(rig.chief.summary().turns).toBe(0);
  });
});
