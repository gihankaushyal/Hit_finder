import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { makeAgentRig, until, AUTH, AGENT_TOKEN, type AgentRig } from "./helpers/agentRig";
import type { AgentEventsResponse, AgentsResponse, ApprovalsResponse } from "../shared/types";

let rig: AgentRig;
beforeEach(async () => {
  rig = await makeAgentRig();
});
afterEach(() => rig.cleanup());

const post = (url: string, body?: unknown) => rig.app.request(url, { method: "POST", headers: AUTH, body: body === undefined ? undefined : JSON.stringify(body) });
const get = (url: string) => rig.app.request(url, { headers: AUTH });

describe("GET /api/agents", () => {
  it("lists agents with the cap and the driver", async () => {
    const res = await get("/api/agents");
    expect(res.status).toBe(200);
    expect((await res.json()) as AgentsResponse).toEqual({ agents: [], max: 3, driver: "fake" });
  });
});

describe("POST /api/agents", () => {
  it("starts a worker", async () => {
    const res = await post("/api/agents", { name: "route-one", task: "tidy the notes", issue: 12 });
    expect(res.status).toBe(201);
    expect(await res.json()).toMatchObject({ name: "route-one", role: "worker", issue: 12, branch: "phase-05-agent-route-one" });
    const list = (await (await get("/api/agents")).json()) as AgentsResponse;
    expect(list.agents.map((a) => a.name)).toEqual(["route-one"]);
  });

  it.each([
    [{ name: "Bad Name", task: "t" }, "name"],
    [{ name: "good-name", task: "" }, "task"],
    [{ name: "good-name", task: 5 }, "task"],
    [{ name: 5, task: "t" }, "name"],
    [{ name: "good-name", task: "t", issue: "7" }, "issue"],
    [{ name: "good-name", task: "t", issue: 0 }, "issue"],
  ] as [Record<string, unknown>, string][])("rejects %j with 400", async (body) => {
    const res = await post("/api/agents", body);
    expect(res.status).toBe(400);
    expect(rig.sup.list()).toEqual([]);
  });

  it("rejects a body that is not a JSON object", async () => {
    expect((await post("/api/agents", [1, 2])).status).toBe(400);
    const res = await rig.app.request("/api/agents", { method: "POST", headers: AUTH, body: "{not json" });
    expect(res.status).toBe(400);
  });

  it("rejects an oversized body", async () => {
    const res = await post("/api/agents", { name: "big-one", task: "x".repeat(70_000) });
    expect(res.status).toBe(400);
  });

  it("answers 409 for a duplicate name and for the cap", async () => {
    expect((await post("/api/agents", { name: "dupe-one", task: "t" })).status).toBe(201);
    expect((await post("/api/agents", { name: "dupe-one", task: "t" })).status).toBe(409);
    expect((await post("/api/agents", { name: "dupe-two", task: "t" })).status).toBe(201);
    expect((await post("/api/agents", { name: "dupe-three", task: "t" })).status).toBe(201);
    const res = await post("/api/agents", { name: "dupe-four", task: "t" });
    expect(res.status).toBe(409);
    expect(((await res.json()) as { error: string }).error).toMatch(/3 agents/);
  });
});

describe("messages, stop and resume", () => {
  it("sends a message to an idle agent", async () => {
    await post("/api/agents", { name: "talker", task: "t" });
    await until(() => rig.sup.get("talker")?.state === "idle" || undefined);
    const res = await post("/api/agents/talker/message", { text: "next please" });
    expect(res.status).toBe(200);
    await until(() => rig.sup.get("talker")?.turns === 2 || undefined);
  });

  it("validates the message and the id", async () => {
    await post("/api/agents", { name: "talker", task: "t" });
    expect((await post("/api/agents/talker/message", { text: "" })).status).toBe(400);
    expect((await post("/api/agents/talker/message", { text: 5 })).status).toBe(400);
    expect((await post("/api/agents/nobody-here/message", { text: "hi" })).status).toBe(404);
  });

  it("stops an agent, 404 for an unknown one, 409 to message a stopped one", async () => {
    await post("/api/agents", { name: "stopper", task: "t" });
    expect((await post("/api/agents/stopper/stop")).status).toBe(200);
    expect(rig.sup.get("stopper")?.state).toBe("stopped");
    expect((await post("/api/agents/stopper/message", { text: "hi" })).status).toBe(409);
    expect((await post("/api/agents/nobody-here/stop")).status).toBe(404);
  });

  it("resumes a stopped agent", async () => {
    await post("/api/agents", { name: "comeback", task: "t" });
    await until(() => rig.sup.get("comeback")?.state === "idle" || undefined);
    await post("/api/agents/comeback/stop");
    expect((await post("/api/agents/comeback/resume")).status).toBe(200);
    await until(() => rig.sup.get("comeback")?.state === "idle" || undefined);
  });
});

describe("GET /api/agents/:id/events", () => {
  it("returns the journal by cursor", async () => {
    await post("/api/agents", { name: "logger", task: "t" });
    await until(() => rig.sup.get("logger")?.state === "idle" || undefined);
    const all = (await (await get("/api/agents/logger/events")).json()) as AgentEventsResponse;
    expect(all.events[0]).toMatchObject({ kind: "user", text: "t" });
    const rest = (await (await get(`/api/agents/logger/events?after=${all.events[0].seq}&limit=2`)).json()) as AgentEventsResponse;
    expect(rest.events).toHaveLength(2);
    expect(rest.events[0].seq).toBe(all.events[1].seq);
    expect(rest.lastSeq).toBe(all.lastSeq);
  });

  it("rejects bad cursors and unknown agents", async () => {
    await post("/api/agents", { name: "logger", task: "t" });
    expect((await get("/api/agents/logger/events?after=-1")).status).toBe(400);
    expect((await get("/api/agents/logger/events?after=abc")).status).toBe(400);
    expect((await get("/api/agents/logger/events?limit=0")).status).toBe(400);
    expect((await get("/api/agents/nobody-here/events")).status).toBe(404);
  });
});

describe("approvals", () => {
  it("lists and decides a pending approval", async () => {
    await post("/api/agents", { name: "pusher", task: "push the branch" });
    await until(() => rig.sup.get("pusher")?.state === "waiting-approval" || undefined);
    const listed = (await (await get("/api/approvals")).json()) as ApprovalsResponse;
    expect(listed.approvals).toHaveLength(1);
    expect(listed.approvals[0]).toMatchObject({ agentName: "pusher", tool: "Bash", input: { command: "git push origin HEAD" } });
    const res = await post(`/api/approvals/${listed.approvals[0].id}/decision`, { decision: "allow" });
    expect(res.status).toBe(200);
    await until(() => rig.sup.get("pusher")?.state === "idle" || undefined);
    expect(((await (await get("/api/approvals")).json()) as ApprovalsResponse).approvals).toEqual([]);
  });

  it("validates decisions and answers 404 for an unknown or repeated id", async () => {
    await post("/api/agents", { name: "pusher", task: "push the branch" });
    await until(() => rig.sup.get("pusher")?.state === "waiting-approval" || undefined);
    const id = rig.approvals.pending()[0].id;
    expect((await post(`/api/approvals/${id}/decision`, { decision: "maybe" })).status).toBe(400);
    expect((await post(`/api/approvals/${id}/decision`, { decision: "deny", message: 5 })).status).toBe(400);
    expect((await post(`/api/approvals/${id}/decision`, { decision: "deny", message: "x".repeat(600) })).status).toBe(400);
    expect(rig.approvals.count()).toBe(1);
    expect((await post(`/api/approvals/${id}/decision`, { decision: "deny", message: "Not now" })).status).toBe(200);
    expect((await post(`/api/approvals/${id}/decision`, { decision: "allow" })).status).toBe(404);
    expect((await post("/api/approvals/not-an-id/decision", { decision: "allow" })).status).toBe(404);
  });
});

describe("access", () => {
  it.each([
    ["GET", "/api/agents"],
    ["POST", "/api/agents"],
    ["POST", "/api/agents/x/message"],
    ["POST", "/api/agents/x/stop"],
    ["POST", "/api/agents/x/resume"],
    ["GET", "/api/agents/x/events"],
    ["GET", "/api/approvals"],
    ["POST", "/api/approvals/x/decision"],
  ])("%s %s without a token is 401", async (method, url) => {
    const res = await rig.app.request(url, { method });
    expect(res.status).toBe(401);
    expect(rig.sup.list()).toEqual([]);
  });

  it("refuses a cross-origin approval even with the token", async () => {
    await post("/api/agents", { name: "pusher", task: "push the branch" });
    await until(() => rig.sup.get("pusher")?.state === "waiting-approval" || undefined);
    const id = rig.approvals.pending()[0].id;
    const cookie = { Cookie: `dash_token=${AGENT_TOKEN}`, "Content-Type": "application/json", Origin: "https://evil.example" };
    const res = await rig.app.request(`http://127.0.0.1:4317/api/approvals/${id}/decision`, { method: "POST", headers: cookie, body: JSON.stringify({ decision: "allow" }) });
    expect(res.status).toBe(403);
    expect(rig.approvals.count()).toBe(1);
  });

  it("never reveals server paths in an error", async () => {
    const res = await post("/api/agents", { name: "good-name", task: "t" });
    const again = await post("/api/agents", { name: "good-name", task: "t" });
    expect(await again.text()).not.toContain(rig.root);
    expect(res.status).toBe(201);
  });
});
