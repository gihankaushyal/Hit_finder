import type { Hono } from "hono";
import type { Gh } from "../lib/gh";
import { parsePrBody } from "../lib/prBody";
import { respondWithError } from "../lib/routeErrors";
import type { PrDetail, PrSummary, PrsResponse } from "../../shared/types";

const LIST_LIMIT = "15";
const RECENT_COUNT = 4;
const SUMMARY_FIELDS = "number,title,state,mergedAt,headRefName,baseRefName,additions,deletions,changedFiles,url";

interface PrViewRaw extends PrSummary {
  body: string;
  files: { path: string; additions: number; deletions: number }[];
}

export function prRoutes(api: Hono, d: { gh: Gh }): void {
  api.get("/prs", async (c) => {
    try {
      const all = await d.gh.json<PrSummary[]>(["pr", "list", "--state", "all", "--limit", LIST_LIMIT, "--json", SUMMARY_FIELDS]);
      const open = all.filter((p) => p.state === "OPEN");
      const merged = all
        .filter((p) => p.state === "MERGED" && p.mergedAt)
        .sort((a, b) => (b.mergedAt! < a.mergedAt! ? -1 : b.mergedAt! > a.mergedAt! ? 1 : 0));
      let latest: PrDetail | null = null;
      if (merged.length > 0) {
        const raw = await d.gh.json<PrViewRaw>(["pr", "view", String(merged[0].number), "--json", `${SUMMARY_FIELDS},body,files`]);
        const { body, files, ...rest } = raw;
        const parsed = parsePrBody(body);
        latest = { ...rest, summary: parsed.summary, testPlan: parsed.testPlan, files: files ?? [] };
      }
      const res: PrsResponse = { latest, recent: merged.slice(1, 1 + RECENT_COUNT), open };
      return c.json(res);
    } catch (err) {
      return respondWithError(c, err);
    }
  });
}
