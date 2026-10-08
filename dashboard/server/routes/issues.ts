import type { Hono } from "hono";
import type { EventBus } from "../events";
import type { Gh } from "../lib/gh";
import { respondWithError } from "../lib/routeErrors";
import type { Issue } from "../../shared/types";

const LIST_LIMIT = "100";
const ISSUE_FIELDS = "number,title,labels,createdAt,updatedAt,url";
const TASK_LABEL = "task";
const TITLE_MAX_CHARS = 200;

interface RawIssue extends Omit<Issue, "labels"> {
  labels: { name: string }[];
}

export function issueRoutes(api: Hono, d: { gh: Gh; bus: EventBus }): void {
  api.get("/issues", async (c) => {
    try {
      const raw = await d.gh.json<RawIssue[]>(["issue", "list", "--state", "open", "--limit", LIST_LIMIT, "--json", ISSUE_FIELDS]);
      const issues: Issue[] = raw
        .map((i) => ({ ...i, labels: i.labels.map((l) => l.name) }))
        .filter((i) => !i.labels.includes(TASK_LABEL));
      return c.json(issues);
    } catch (err) {
      return respondWithError(c, err);
    }
  });

  api.post("/issues", async (c) => {
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
    if (body !== undefined && typeof body !== "string") return c.json({ error: "body must be a string" }, 400);
    try {
      const url = await d.gh.text(["issue", "create", "--title", trimmed, "--body", body ?? ""]);
      d.bus.emit({ type: "invalidate", resource: "issues" });
      return c.json({ url }, 201);
    } catch (err) {
      return respondWithError(c, err);
    }
  });
}
