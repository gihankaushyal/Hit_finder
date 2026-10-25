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
  name: string;
  description: string;
  art: string | null;
}

async function readSkills(skillsDir: string): Promise<SkillFile[]> {
  let entries: fs.Dirent[];
  try {
    entries = await fs.promises.readdir(skillsDir, { withFileTypes: true });
  } catch {
    return [];
  }
  const skills: SkillFile[] = [];
  // Folder order decides which of two same-named skills wins, so do not leave it to the file system.
  for (const e of entries.sort((a, b) => a.name.localeCompare(b.name))) {
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
        name: typeof fm.name === "string" && fm.name ? fm.name : e.name,
        description: typeof fm.description === "string" ? fm.description : "",
        art: typeof fm[ART_KEY] === "string" && fm[ART_KEY] ? fm[ART_KEY] : null,
      });
    } catch {
      // An unreadable header must not hide the skill.
      skills.push({ name: e.name, description: "The header of this SKILL.md could not be read.", art: null });
    }
  }
  // Two folders sharing a name would clash on the card list, so the first one wins.
  const seen = new Set<string>();
  return skills.sort((a, b) => a.name.localeCompare(b.name)).filter((s) => !seen.has(s.name) && !!seen.add(s.name));
}

export function skillRoutes(api: Hono, d: { config: Config }): void {
  api.get("/skills", async (c) => {
    const dirs = d.config.transcriptsIncludeWorktrees ? await withWorktreeDirs(d.config.transcriptsDir) : [d.config.transcriptsDir];
    const [skills, runs] = await Promise.all([readSkills(d.config.skillsDir), scanLastRuns(dirs)]);
    const body: SkillsResponse = {
      skills: skills.map((s): SkillSummary => ({ name: s.name, description: s.description, lastRun: runs.get(s.name) ?? null, art: s.art })),
    };
    return c.json(body);
  });
}
