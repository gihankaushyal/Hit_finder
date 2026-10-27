import type { Context, Hono } from "hono";
import type { Chief } from "../agents/chief";
import { CONV_ID } from "../agents/conversations";
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
  /** The `:id` of a conversation route, or a 400 for anything that is not a conversation id. */
  const convId = (c: Context): string | Response => {
    const id = c.req.param("id") ?? "";
    return CONV_ID.test(id) ? id : badRequest(c, "invalid conversation id");
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

  // Conversations: `?archived=1` lists only archived ones, `?archived=0` only the others, no query lists all.
  api.get("/chief/conversations", (c) => {
    const q = c.req.query("archived");
    const rows = d.chief.conversations().filter((r) => (q === "1" ? r.archived : q === "0" ? !r.archived : true));
    return c.json({ conversations: rows });
  });
  api.post("/chief/conversations", (c) => done(c, () => d.chief.reset()));

  api.post("/chief/conversations/:id/resume", (c) => {
    const id = convId(c);
    return typeof id === "string" ? done(c, () => d.chief.resume(id)) : id;
  });

  api.patch("/chief/conversations/:id", limit, async (c) => {
    const id = convId(c);
    if (typeof id !== "string") return id;
    const input = await readObject(c);
    if (!input) return badRequest(c, "body must be a JSON object");
    const { title, archived } = input;
    if (title !== undefined && typeof title !== "string") return badRequest(c, "title must be a string");
    if (archived !== undefined && typeof archived !== "boolean") return badRequest(c, "archived must be true or false");
    if (title === undefined && archived === undefined) return badRequest(c, "nothing to change");
    return done(c, () => d.chief.patch(id, { title: title as string | undefined, archived: archived as boolean | undefined }));
  });

  api.get("/chief/conversations/:id/events", async (c) => {
    const id = convId(c);
    if (typeof id !== "string") return id;
    const q = eventsQuery(c);
    if (q instanceof Response) return q;
    try {
      return c.json(await d.chief.eventsFor(id, q.after, q.limit));
    } catch (err) {
      return failAgent(c, err);
    }
  });
}
