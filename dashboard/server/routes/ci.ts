import type { Hono } from "hono";
import type { Gh } from "../lib/gh";
import { respondWithError } from "../lib/routeErrors";
import type { CiRun } from "../../shared/types";

const LIST_LIMIT = "40";
const RETURN_COUNT = 10;
const CI_WORKFLOW_NAME = "CI";
const RUN_FIELDS = "databaseId,name,status,conclusion,headBranch,event,createdAt,url";

interface RawRun {
  databaseId: number; name: string; status: string; conclusion: string | null;
  headBranch: string; event: string; createdAt: string; url: string;
}

export function ciRoutes(api: Hono, d: { gh: Gh }): void {
  api.get("/ci", async (c) => {
    try {
      const raw = await d.gh.json<RawRun[]>(["run", "list", "--limit", LIST_LIMIT, "--json", RUN_FIELDS]);
      const runs: CiRun[] = raw
        .filter((r) => r.name === CI_WORKFLOW_NAME)
        .slice(0, RETURN_COUNT)
        .map((r) => ({
          id: r.databaseId, status: r.status, conclusion: r.conclusion,
          branch: r.headBranch, event: r.event, createdAt: r.createdAt, url: r.url,
        }));
      return c.json({ runs });
    } catch (err) {
      return respondWithError(c, err);
    }
  });
}
