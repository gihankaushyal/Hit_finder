import fs from "node:fs";
import path from "node:path";
import type { Hono } from "hono";
import type { SkillSummary, SkillsResponse } from "../../shared/types";
import type { Config } from "../config";
import { scanLastRuns } from "../skills/lastRun";

/** Skills that submit jobs or open PRs; the UI asks for a second click before starting them. */
const CONFIRM_SKILLS = new Set(["slurm-job", "docs-refresh"]);
const SKILL_FILE = "SKILL.md";
const FRONTMATTER_RE = /^---\r?\n([\s\S]*?)\r?\n---/;

/** `name` and `description` from the SKILL.md frontmatter (single-line values, as the project's skills use). */
function frontmatter(text: string): { name?: string; description?: string } {
  const block = FRONTMATTER_RE.exec(text)?.[1] ?? "";
  const out: Record<string, string> = {};
  for (const line of block.split(/\r?\n/)) {
    const m = /^(name|description):\s*(.*)$/.exec(line);
    if (m) out[m[1]] = m[2].trim();
  }
  return out;
}

async function readSkills(dir: string): Promise<Array<{ name: string; description: string }>> {
  let entries: fs.Dirent[];
  try {
    entries = await fs.promises.readdir(dir, { withFileTypes: true });
  } catch {
    return [];
  }
  const skills: Array<{ name: string; description: string }> = [];
  for (const e of entries) {
    if (!e.isDirectory()) continue;
    try {
      const fm = frontmatter(await fs.promises.readFile(path.join(dir, e.name, SKILL_FILE), "utf8"));
      skills.push({ name: fm.name ?? e.name, description: fm.description ?? "" });
    } catch {
      // a folder without a SKILL.md is not a skill
    }
  }
  return skills.sort((a, b) => a.name.localeCompare(b.name));
}

export function skillRoutes(api: Hono, d: { config: Config }): void {
  api.get("/skills", async (c) => {
    const [skills, runs] = await Promise.all([readSkills(d.config.skillsDir), scanLastRuns(d.config.transcriptsDir)]);
    const body: SkillsResponse = {
      skills: skills.map((s): SkillSummary => ({ ...s, lastRun: runs.get(s.name) ?? null, confirmRequired: CONFIRM_SKILLS.has(s.name) })),
    };
    return c.json(body);
  });
}
