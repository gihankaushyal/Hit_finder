import type { Context, Hono } from "hono";
import { bodyLimit } from "hono/body-limit";
import type { AgentsResponse } from "../../shared/types";
import { AgentError, type AgentErrorCode, type Supervisor } from "../agents/supervisor";
import type { Approvals } from "../agents/approvals";
import type { Config } from "../config";
import { respondWithError } from "../lib/routeErrors";

const REQUEST_MAX_BYTES = 64 * 1024;
const EVENTS_DEFAULT_LIMIT = 200;
const EVENTS_MAX_LIMIT = 500;
const DECISION_MESSAGE_MAX_CHARS = 500;
const WHOLE_NUMBER = /^\d{1,12}$/;

const STATUS: Record<AgentErrorCode, 400 | 404 | 409> = { invalid: 400, "not-found": 404, cap: 409, exists: 409, state: 409 };

async function readObject(c: Context): Promise<Record<string, unknown> | null> {
  try {
    const v: unknown = await c.req.json();
    return typeof v === "object" && v !== null && !Array.isArray(v) ? (v as Record<string, unknown>) : null;
  } catch {
    return null;
  }
}

export function agentRoutes(api: Hono, d: { agents: Supervisor; approvals: Approvals; config: Config }): void {
  const limit = bodyLimit({ maxSize: REQUEST_MAX_BYTES, onError: (c) => c.json({ error: "request body must be at most 64 KB" }, 400) });
  const fail = (c: Context, err: unknown) => (err instanceof AgentError ? c.json({ error: err.message }, STATUS[err.code]) : respondWithError(c, err));
  const bad = (c: Context, message: string) => c.json({ error: message }, 400);

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
    const afterRaw = c.req.query("after") ?? "0";
    const limitRaw = c.req.query("limit") ?? String(EVENTS_DEFAULT_LIMIT);
    if (!WHOLE_NUMBER.test(afterRaw)) return bad(c, "after must be a whole number");
    const n = Number(limitRaw);
    if (!WHOLE_NUMBER.test(limitRaw) || n < 1 || n > EVENTS_MAX_LIMIT) return bad(c, `limit must be 1 to ${EVENTS_MAX_LIMIT}`);
    try {
      return c.json(await d.agents.events(c.req.param("id"), Number(afterRaw), n));
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
