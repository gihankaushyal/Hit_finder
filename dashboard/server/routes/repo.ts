import type { Hono } from "hono";
import { respondWithError } from "../lib/routeErrors";
import type { Runner } from "../runner";

async function git(runner: Runner, repoRoot: string, args: string[]): Promise<string> {
  const r = await runner("git", args, { cwd: repoRoot });
  if (r.code !== 0) throw new Error(`git ${args.join(" ")} failed (exit ${r.code}): ${r.stderr.trim().slice(0, 300)}`);
  return r.stdout.trim();
}

const SHORT_HASH_LENGTH = 7;

/** GET /repo: the current branch and commit (short for display, full for copying), for the top bar. */
export function repoRoutes(api: Hono, d: { runner: Runner; repoRoot: string }): void {
  api.get("/repo", async (c) => {
    try {
      // One call, so the hash and the branch come from the same HEAD: --abbrev-ref applies to the arguments after it.
      const out = await git(d.runner, d.repoRoot, ["rev-parse", "HEAD", "--abbrev-ref", "HEAD"]);
      const [commitFull, branch] = out.split("\n").map((l) => l.trim());
      if (!commitFull || !branch) throw new Error(`git rev-parse gave unexpected output: ${out.slice(0, 100)}`);
      return c.json({ branch, commit: commitFull.slice(0, SHORT_HASH_LENGTH), commitFull });
    } catch (err) {
      return respondWithError(c, err);
    }
  });
}
