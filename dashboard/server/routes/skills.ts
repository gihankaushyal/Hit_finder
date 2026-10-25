import fs from "node:fs";
import path from "node:path";
import type { Hono } from "hono";
import type { SkillSummary, SkillsResponse } from "../../shared/types";
import type { Config } from "../config";
import { parseFrontmatter } from "../skills/frontmatter";
import { scanLastRuns, withWorktreeDirs } from "../skills/lastRun";

const SKILL_FILE = "SKILL.md";
/** SKILL.md header key that picks the cartoon. */
const ART_KEY = "dashboard-art";

interface SkillFile {
  /** The folder under .claude/skills; breaks ties between two skills that share a name. */
  dir: string;
  name: string;
  description: string;
  art: string | null;
  /** False when the header could not be parsed, so the skill may not be invocable. */
  readable: boolean;
}

async function readSkills(skillsDir: string): Promise<SkillFile[]> {
  let entries: fs.Dirent[];
  try {
    entries = await fs.promises.readdir(skillsDir, { withFileTypes: true });
  } catch {
    return [];
  }
  const skills: SkillFile[] = [];
  for (const e of entries) {
    if (!e.isDirectory()) continue;
    let text: string;
    try {
      text = await fs.promises.readFile(path.join(skillsDir, e.name, SKILL_FILE), "utf8");
    } catch {
      continue; // a folder without a SKILL.md is not a skill
    }
    try {
      const fm = parseFrontmatter(text);
      skills.push({
        dir: e.name,
        name: typeof fm.name === "string" && fm.name ? fm.name : e.name,
        description: typeof fm.description === "string" ? fm.description : "",
        art: typeof fm[ART_KEY] === "string" && fm[ART_KEY] ? fm[ART_KEY] : null,
        readable: true,
      });
    } catch {
      // An unreadable header must not hide the skill.
      skills.push({ dir: e.name, name: e.name, description: "The header of this SKILL.md could not be read.", art: null, readable: false });
    }
  }
  // Two folders sharing a name would clash on the card list, so the first one wins.
  const seen = new Set<string>();
  return skills.sort((a, b) => a.name.localeCompare(b.name) || a.dir.localeCompare(b.dir)).filter((s) => !seen.has(s.name) && !!seen.add(s.name));
}

export function skillRoutes(api: Hono, d: { config: Config }): void {
  api.get("/skills", async (c) => {
    const dirs = d.config.transcriptsIncludeWorktrees ? await withWorktreeDirs(d.config.transcriptsDir) : [d.config.transcriptsDir];
    const [skills, runs] = await Promise.all([readSkills(d.config.skillsDir), scanLastRuns(dirs)]);
    const body: SkillsResponse = {
      skills: skills.map((s): SkillSummary => ({ name: s.name, description: s.description, lastRun: runs.get(s.name) ?? null, art: s.art, readable: s.readable })),
    };
    return c.json(body);
  });
}
