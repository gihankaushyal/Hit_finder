import type { Hono } from "hono";
import { respondWithError } from "../lib/routeErrors";
import type { Runner } from "../runner";

async function git(runner: Runner, repoRoot: string, args: string[]): Promise<string> {
  const r = await runner("git", args, { cwd: repoRoot });
  if (r.code !== 0) throw new Error(`git ${args.join(" ")} failed (exit ${r.code}): ${r.stderr.trim().slice(0, 300)}`);
  return r.stdout.trim();
}

/** GET /repo: the current branch and short commit, for the top bar. */
export function repoRoutes(api: Hono, d: { runner: Runner; repoRoot: string }): void {
  api.get("/repo", async (c) => {
    try {
      const branch = await git(d.runner, d.repoRoot, ["rev-parse", "--abbrev-ref", "HEAD"]);
      const commit = await git(d.runner, d.repoRoot, ["rev-parse", "--short", "HEAD"]);
      return c.json({ branch, commit });
    } catch (err) {
      return respondWithError(c, err);
    }
  });
}
