import type { Context } from "hono";
import { bodyLimit } from "hono/body-limit";
import { AgentError, type AgentErrorCode } from "../agents/supervisor";
import { respondWithError } from "./routeErrors";

export const REQUEST_MAX_BYTES = 64 * 1024;
export const EVENTS_DEFAULT_LIMIT = 200;
export const EVENTS_MAX_LIMIT = 500;
const WHOLE_NUMBER = /^\d{1,12}$/;
const STATUS: Record<AgentErrorCode, 400 | 404 | 409> = { invalid: 400, "not-found": 404, cap: 409, exists: 409, state: 409 };

export const jsonLimit = () => bodyLimit({ maxSize: REQUEST_MAX_BYTES, onError: (c) => c.json({ error: "request body must be at most 64 KB" }, 400) });

export async function readObject(c: Context): Promise<Record<string, unknown> | null> {
  try {
    const v: unknown = await c.req.json();
    return typeof v === "object" && v !== null && !Array.isArray(v) ? (v as Record<string, unknown>) : null;
  } catch {
    return null;
  }
}

export const badRequest = (c: Context, message: string) => c.json({ error: message }, 400);

/** AgentError messages are written to be shown; anything else is logged and answered with a fixed message. */
export const failAgent = (c: Context, err: unknown) => (err instanceof AgentError ? c.json({ error: err.message }, STATUS[err.code]) : respondWithError(c, err));

/** Parses `?after=&limit=` for a journal read; returns a response when the query is invalid. */
export function eventsQuery(c: Context): { after: number; limit: number } | Response {
  const afterRaw = c.req.query("after") ?? "0";
  const limitRaw = c.req.query("limit") ?? String(EVENTS_DEFAULT_LIMIT);
  if (!WHOLE_NUMBER.test(afterRaw)) return badRequest(c, "after must be a whole number");
  const limit = Number(limitRaw);
  if (!WHOLE_NUMBER.test(limitRaw) || limit < 1 || limit > EVENTS_MAX_LIMIT) return badRequest(c, `limit must be 1 to ${EVENTS_MAX_LIMIT}`);
  return { after: Number(afterRaw), limit };
}
