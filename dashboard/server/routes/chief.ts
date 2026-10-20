import type { Context, Hono } from "hono";
import type { Chief } from "../agents/chief";
import { badRequest, eventsQuery, failAgent, jsonLimit, readObject } from "../lib/http";

export function chiefRoutes(api: Hono, d: { chief: Chief }): void {
  const limit = jsonLimit();
  const done = async (c: Context, f: () => Promise<void>) => {
    try {
      await f();
      return c.json({ ok: true });
    } catch (err) {
      return failAgent(c, err);
    }
  };

  api.get("/chief", (c) => c.json(d.chief.summary()));

  api.post("/chief/message", limit, async (c) => {
    const input = await readObject(c);
    if (!input) return badRequest(c, "body must be a JSON object");
    if (typeof input.text !== "string") return badRequest(c, "text must be a string");
    return done(c, () => d.chief.send(input.text as string));
  });

  api.get("/chief/events", async (c) => {
    const q = eventsQuery(c);
    if (q instanceof Response) return q;
    try {
      return c.json(await d.chief.events(q.after, q.limit));
    } catch (err) {
      return failAgent(c, err);
    }
  });

  api.post("/chief/stop", (c) => done(c, () => d.chief.stop()));
  api.post("/chief/reset", (c) => done(c, () => d.chief.reset()));
}
