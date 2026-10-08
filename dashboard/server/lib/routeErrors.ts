import type { Context } from "hono";
import { GhError } from "./gh";

const INTERNAL_ERROR = "internal error";

/** Log the full error server-side; send the client only a fixed safe message. */
export function respondWithError(c: Context, err: unknown): Response {
  console.error("request failed:", err);
  if (err instanceof GhError) return c.json({ error: err.publicMessage }, 502);
  return c.json({ error: INTERNAL_ERROR }, 500);
}
