import type { Hono } from "hono";
import type { AgentsResponse } from "../../shared/types";
import { AgentError, type Supervisor } from "../agents/supervisor";
import type { Approvals } from "../agents/approvals";
import type { Config } from "../config";
import { badRequest, eventsQuery, failAgent, jsonLimit, readObject } from "../lib/http";

const DECISION_MESSAGE_MAX_CHARS = 500;

export function agentRoutes(api: Hono, d: { agents: Supervisor; approvals: Approvals; config: Config }): void {
  const limit = jsonLimit();
  const fail = failAgent;
  const bad = badRequest;

  api.get("/agents", (c) => {
    const body: AgentsResponse = { agents: d.agents.list(), max: d.config.agentMax, driver: d.config.agentDriver };
    return c.json(body);
  });

  api.post("/agents", limit, async (c) => {
    const input = await readObject(c);
    if (!input) return bad(c, "body must be a JSON object");
    const { name, task, issue } = input;
    if (typeof name !== "string") return bad(c, "name must be a string");
    if (typeof task !== "string") return bad(c, "task must be a string");
    if (issue !== undefined && issue !== null && (typeof issue !== "number" || !Number.isInteger(issue) || issue < 1)) return bad(c, "issue must be a positive whole number");
    try {
      return c.json(await d.agents.spawn({ name, task, issue: (issue as number | null | undefined) ?? null }), 201);
    } catch (err) {
      return fail(c, err);
    }
  });

  api.post("/agents/:id/message", limit, async (c) => {
    const input = await readObject(c);
    if (!input) return bad(c, "body must be a JSON object");
    if (typeof input.text !== "string") return bad(c, "text must be a string");
    try {
      await d.agents.message(c.req.param("id"), input.text);
      return c.json({ ok: true });
    } catch (err) {
      return fail(c, err);
    }
  });

  api.post("/agents/:id/stop", async (c) => {
    try {
      if (!d.agents.get(c.req.param("id"))) throw new AgentError("not-found", "Agent not found.");
      await d.agents.stop(c.req.param("id"));
      return c.json({ ok: true });
    } catch (err) {
      return fail(c, err);
    }
  });

  api.post("/agents/:id/resume", async (c) => {
    try {
      await d.agents.resume(c.req.param("id"));
      return c.json({ ok: true });
    } catch (err) {
      return fail(c, err);
    }
  });

  api.get("/agents/:id/events", async (c) => {
    const q = eventsQuery(c);
    if (q instanceof Response) return q;
    try {
      return c.json(await d.agents.events(c.req.param("id"), q.after, q.limit));
    } catch (err) {
      return fail(c, err);
    }
  });

  api.get("/approvals", (c) => c.json({ approvals: d.approvals.pending() }));

  api.post("/approvals/:id/decision", limit, async (c) => {
    const input = await readObject(c);
    if (!input) return bad(c, "body must be a JSON object");
    const { decision, message } = input;
    if (decision !== "allow" && decision !== "deny") return bad(c, 'decision must be "allow" or "deny"');
    if (message !== undefined && (typeof message !== "string" || message.length > DECISION_MESSAGE_MAX_CHARS)) return bad(c, `message must be text of at most ${DECISION_MESSAGE_MAX_CHARS} characters`);
    try {
      d.approvals.decide(c.req.param("id"), decision, message as string | undefined);
      return c.json({ ok: true });
    } catch {
      return c.json({ error: "approval not found" }, 404);
    }
  });
}
