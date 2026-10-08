import type { Hono } from "hono";
import type { Gh } from "../lib/gh";
import { parsePrBody } from "../lib/prBody";
import { respondWithError } from "../lib/routeErrors";
import type { PrChecks, PrDetail, PrSummary, PrsResponse } from "../../shared/types";

const LIST_LIMIT = "15";
const RECENT_COUNT = 4;
const SUMMARY_FIELDS = "number,title,state,mergedAt,headRefName,baseRefName,additions,deletions,changedFiles,url";

/** One entry of gh's statusCheckRollup: a CheckRun (status, conclusion) or a StatusContext (state). */
interface RollupEntry {
  status?: string;
  conclusion?: string;
  state?: string;
}
type PrListRaw = PrSummary & { statusCheckRollup?: RollupEntry[] | null };

const FAILING = new Set(["FAILURE", "ERROR", "TIMED_OUT", "CANCELLED", "ACTION_REQUIRED", "STARTUP_FAILURE"]);
const PASSING = new Set(["SUCCESS", "NEUTRAL", "SKIPPED"]);

/** Reduces gh's check list to one state plus counts. Anything not finished or not recognised counts as pending. */
export function reduceChecks(rollup: RollupEntry[] | null | undefined): PrChecks {
  const out: PrChecks = { state: "none", passing: 0, failing: 0, pending: 0 };
  for (const e of rollup ?? []) {
    const verdict = (e.state ?? (e.status === "COMPLETED" ? e.conclusion : undefined) ?? "").toUpperCase();
    if (FAILING.has(verdict)) out.failing++;
    else if (PASSING.has(verdict)) out.passing++;
    else out.pending++;
  }
  out.state = out.failing > 0 ? "failing" : out.pending > 0 ? "pending" : out.passing > 0 ? "passing" : "none";
  return out;
}

interface PrViewRaw extends PrSummary {
  body: string;
  files: { path: string; additions: number; deletions: number }[];
}

export function prRoutes(api: Hono, d: { gh: Gh }): void {
  api.get("/prs", async (c) => {
    try {
      const raw = await d.gh.json<PrListRaw[]>(["pr", "list", "--state", "all", "--limit", LIST_LIMIT, "--json", `${SUMMARY_FIELDS},statusCheckRollup`]);
      const all: PrSummary[] = raw.map(({ statusCheckRollup: _rollup, ...rest }) => rest);
      const open = raw
        .filter((p) => p.state === "OPEN")
        .map(({ statusCheckRollup, ...rest }): PrSummary => ({ ...rest, checks: reduceChecks(statusCheckRollup) }));
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
