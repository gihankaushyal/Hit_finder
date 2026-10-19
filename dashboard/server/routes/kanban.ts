import type { Hono } from "hono";
import { bodyLimit } from "hono/body-limit";
import { respondWithError } from "../lib/routeErrors";
import { NotImportedError, TaskNotFoundError, type KanbanService } from "../lib/kanbanService";
import type { TaskStatus } from "../../shared/types";

const TITLE_MAX_CHARS = 200;
const BODY_MAX_CHARS = 60_000;
const REQUEST_MAX_BYTES = 64 * 1024;
const ISSUE_NUMBER = /^[1-9]\d{0,9}$/;
const STATUSES: readonly TaskStatus[] = ["todo", "in-progress", "blocked", "done"];

export function kanbanRoutes(api: Hono, d: { service: KanbanService }): void {
  const limit = bodyLimit({
    maxSize: REQUEST_MAX_BYTES,
    onError: (c) => c.json({ error: "request body must be at most 64 KB" }, 400),
  });
  const fail = (c: Parameters<typeof respondWithError>[0], err: unknown) => {
    if (err instanceof NotImportedError) return c.json({ error: err.message }, 409);
    if (err instanceof TaskNotFoundError) return c.json({ error: "task not found" }, 404);
    return respondWithError(c, err);
  };
  const readObject = async (c: { req: { json(): Promise<unknown> } }): Promise<Record<string, unknown> | null> => {
    try {
      const v = await c.req.json();
      return typeof v === "object" && v !== null && !Array.isArray(v) ? (v as Record<string, unknown>) : null;
    } catch {
      return null;
    }
  };

  api.get("/kanban", async (c) => {
    try {
      return c.json(await d.service.board());
    } catch (err) {
      return fail(c, err);
    }
  });

  api.post("/kanban/tasks", limit, async (c) => {
    const input = await readObject(c);
    if (!input) return c.json({ error: "body must be a JSON object" }, 400);
    const { title, body } = input;
    if (typeof title !== "string") return c.json({ error: "title must be a string" }, 400);
    const trimmed = title.trim();
    if (trimmed.length < 1 || trimmed.length > TITLE_MAX_CHARS) {
      return c.json({ error: `title must be 1 to ${TITLE_MAX_CHARS} characters` }, 400);
    }
    if (trimmed.includes("\0")) return c.json({ error: "title must not contain NUL characters" }, 400);
    if (body !== undefined && typeof body !== "string") return c.json({ error: "body must be a string" }, 400);
    if (typeof body === "string" && (body.length > BODY_MAX_CHARS || body.includes("\0"))) {
      return c.json({ error: `body must be at most ${BODY_MAX_CHARS} characters and contain no NUL` }, 400);
    }
    try {
      await d.service.addTask(trimmed, body as string | undefined);
      return c.json({ ok: true }, 201);
    } catch (err) {
      return fail(c, err);
    }
  });

  api.patch("/kanban/tasks/:n", limit, async (c) => {
    const raw = c.req.param("n");
    if (!ISSUE_NUMBER.test(raw)) return c.json({ error: "issue number must be a positive integer" }, 400);
    const input = await readObject(c);
    if (!input) return c.json({ error: "body must be a JSON object" }, 400);
    const status = input.status;
    if (typeof status !== "string" || !STATUSES.includes(status as TaskStatus)) {
      return c.json({ error: `status must be one of ${STATUSES.join(", ")}` }, 400);
    }
    try {
      await d.service.setStatus(Number(raw), status as TaskStatus);
      return c.json({ ok: true }, 200);
    } catch (err) {
      return fail(c, err);
    }
  });

  api.post("/kanban/sync", async (c) => {
    try {
      await d.service.sync();
      return c.json(await d.service.board());
    } catch (err) {
      return fail(c, err);
    }
  });
}
