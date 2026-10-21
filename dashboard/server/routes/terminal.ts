import type { Context, Hono } from "hono";
import { stream } from "hono/streaming";
import type { Config } from "../config";
import { badRequest, jsonLimit, readObject } from "../lib/http";
import { respondWithError } from "../lib/routeErrors";
import { TerminalError, type Terminal, type TerminalErrorCode } from "../terminal/terminal";

const STATUS: Record<TerminalErrorCode, 400 | 403 | 404 | 409> = { invalid: 400, disabled: 403, "not-found": 404, limit: 409, state: 409 };
const SAME_SITE_SIGNALS = new Set(["same-origin", "none"]);

export function terminalRoutes(api: Hono, d: { terminal: Terminal; config: Config }): void {
  const limit = jsonLimit();
  const fail = (c: Context, err: unknown) => (err instanceof TerminalError ? c.json({ error: err.message }, STATUS[err.code]) : respondWithError(c, err));
  const guarded = async (c: Context, f: () => void | Promise<void>) => {
    try {
      await f();
      return c.json({ ok: true });
    } catch (err) {
      return fail(c, err);
    }
  };

  api.get("/terminal", (c) => c.json({ enabled: d.terminal.enabled, max: d.config.terminalMax, open: d.terminal.count() }));

  api.post("/terminal/open", limit, async (c) => {
    const input = await readObject(c);
    if (!input) return badRequest(c, "body must be a JSON object");
    if (typeof input.cols !== "number" || typeof input.rows !== "number") return badRequest(c, "cols and rows must be numbers");
    try {
      return c.json(d.terminal.open(input.cols, input.rows), 201);
    } catch (err) {
      return fail(c, err);
    }
  });

  // Attaching starts a shell, so unlike other GETs it must come from this page itself: the browser's
  // own signals are checked even though a cookie alone would otherwise be enough.
  api.get("/terminal/:id/stream", (c) => {
    const origin = c.req.header("Origin");
    const site = c.req.header("Sec-Fetch-Site");
    if ((origin !== undefined && origin !== new URL(c.req.url).origin) || (site !== undefined && !SAME_SITE_SIGNALS.has(site))) {
      return c.json({ error: "cross-site request refused" }, 403);
    }
    const id = c.req.param("id");
    let output;
    try {
      output = d.terminal.attach(id).output;
    } catch (err) {
      return fail(c, err);
    }
    c.header("Content-Type", "application/octet-stream");
    c.header("Cache-Control", "no-store");
    c.header("X-Accel-Buffering", "no");
    return stream(c, async (s) => {
      s.onAbort(() => void d.terminal.close(id));
      try {
        for await (const chunk of output) {
          if (s.aborted) break;
          await s.write(chunk as Uint8Array);
        }
      } catch {
        // the browser went away mid-write
      } finally {
        void d.terminal.close(id);
      }
    });
  });

  api.post("/terminal/:id/input", limit, async (c) => {
    const input = await readObject(c);
    if (!input) return badRequest(c, "body must be a JSON object");
    if (typeof input.data !== "string") return badRequest(c, "data must be a string");
    return guarded(c, () => d.terminal.input(c.req.param("id"), input.data as string));
  });

  api.post("/terminal/:id/resize", limit, async (c) => {
    const input = await readObject(c);
    if (!input) return badRequest(c, "body must be a JSON object");
    if (typeof input.cols !== "number" || typeof input.rows !== "number") return badRequest(c, "cols and rows must be numbers");
    return guarded(c, () => d.terminal.resize(c.req.param("id"), input.cols as number, input.rows as number));
  });

  api.delete("/terminal/:id", (c) => guarded(c, () => d.terminal.close(c.req.param("id"))));
}
