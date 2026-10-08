import type { Hono } from "hono";
import { bodyLimit } from "hono/body-limit";
import type { EventBus } from "../events";
import { flagArg, type Gh } from "../lib/gh";
import { respondWithError } from "../lib/routeErrors";
import type { Issue } from "../../shared/types";

const LIST_LIMIT = "100";
const ISSUE_FIELDS = "number,title,labels,createdAt,updatedAt,url";
const TASK_LABEL = "task";
const TITLE_MAX_CHARS = 200;
const BODY_MAX_CHARS = 60_000;
const REQUEST_MAX_BYTES = 64 * 1024;
// Excluded in the query so task issues cannot push ordinary ones out of the limit window.
const EXCLUDE_TASKS_SEARCH = `--search=-label:${TASK_LABEL}`;

interface RawIssue extends Omit<Issue, "labels"> {
  labels: { name: string }[];
}

export function issueRoutes(api: Hono, d: { gh: Gh; bus: EventBus }): void {
  api.get("/issues", async (c) => {
    try {
      const raw = await d.gh.json<RawIssue[]>(["issue", "list", "--state", "open", "--limit", LIST_LIMIT, EXCLUDE_TASKS_SEARCH, "--json", ISSUE_FIELDS]);
      const issues: Issue[] = raw
        .map((i) => ({ ...i, labels: i.labels.map((l) => l.name) }))
        .filter((i) => !i.labels.includes(TASK_LABEL));
      return c.json(issues);
    } catch (err) {
      return respondWithError(c, err);
    }
  });

  const limit = bodyLimit({
    maxSize: REQUEST_MAX_BYTES,
    onError: (c) => c.json({ error: "request body must be at most 64 KB" }, 400),
  });
  api.post("/issues", limit, async (c) => {
    let input: unknown;
    try {
      input = await c.req.json();
    } catch {
      return c.json({ error: "invalid JSON body" }, 400);
    }
    if (typeof input !== "object" || input === null || Array.isArray(input)) {
      return c.json({ error: "body must be an object" }, 400);
    }
    const { title, body } = input as { title?: unknown; body?: unknown };
    if (typeof title !== "string") return c.json({ error: "title must be a string" }, 400);
    const trimmed = title.trim();
    if (trimmed.length < 1 || trimmed.length > TITLE_MAX_CHARS) {
      return c.json({ error: `title must be 1 to ${TITLE_MAX_CHARS} characters` }, 400);
    }
    if (trimmed.includes("\0")) return c.json({ error: "title must not contain NUL characters" }, 400);
    if (body !== undefined && typeof body !== "string") return c.json({ error: "body must be a string" }, 400);
    if (body !== undefined && body.length > BODY_MAX_CHARS) {
      return c.json({ error: `body must be at most ${BODY_MAX_CHARS} characters` }, 400);
    }
    if (body !== undefined && body.includes("\0")) return c.json({ error: "body must not contain NUL characters" }, 400);
    try {
      const url = await d.gh.text(["issue", "create", flagArg("title", trimmed), flagArg("body", body ?? "")]);
      d.bus.emit({ type: "invalidate", resource: "issues" });
      return c.json({ url }, 201);
    } catch (err) {
      return respondWithError(c, err);
    }
  });
}
