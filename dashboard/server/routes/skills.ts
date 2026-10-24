import { randomBytes } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import type { Hono } from "hono";
import type { SkillSummary, SkillsResponse } from "../../shared/types";
import { AgentError, type Supervisor } from "../agents/supervisor";
import type { Config } from "../config";
import { badRequest, failAgent, jsonLimit, readObject } from "../lib/http";
import { parseFrontmatter } from "../skills/frontmatter";
import { scanLastRuns, withWorktreeDirs } from "../skills/lastRun";

const SKILL_FILE = "SKILL.md";
/** SKILL.md header keys the dashboard reads: a second click before running, and which cartoon to show. */
const CONFIRM_KEY = "dashboard-confirm";
const ART_KEY = "dashboard-art";
const ARGS_MAX = 500;
const SLUG_MAX = 14;
const NAME_TRIES = 5;
const SUFFIX_BYTES = 2;

interface SkillFile {
  /** The folder under .claude/skills, which is where SKILL.md lives. */
  dir: string;
  name: string;
  description: string;
  confirmRequired: boolean;
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
        confirmRequired: fm[CONFIRM_KEY] === true,
        art: typeof fm[ART_KEY] === "string" && fm[ART_KEY] ? fm[ART_KEY] : null,
      });
    } catch {
      // An unreadable header must not hide the skill or quietly drop its confirm step.
      skills.push({ dir: e.name, name: e.name, description: "The header of this SKILL.md could not be read.", confirmRequired: true, art: null });
    }
  }
  return skills.sort((a, b) => a.name.localeCompare(b.name));
}

/** Lower-case letters, digits and single hyphens, short enough that `sk-<slug>-<4 hex>` fits the 31-character agent name limit. */
function slugOf(name: string): string {
  const slug = name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, SLUG_MAX).replace(/-+$/, "");
  return slug || "skill";
}

function taskFor(s: SkillFile, args: string): string {
  const base = `Run the project skill "${s.name}". Read .claude/skills/${s.dir}/${SKILL_FILE} in this repository and follow its instructions.`;
  return args ? `${base}\n\nArguments: ${args}` : base;
}

export function skillRoutes(api: Hono, d: { config: Config; agents: Supervisor }): void {
  const limit = jsonLimit();

  api.get("/skills", async (c) => {
    const dirs = d.config.transcriptsIncludeWorktrees ? await withWorktreeDirs(d.config.transcriptsDir) : [d.config.transcriptsDir];
    const [skills, runs] = await Promise.all([readSkills(d.config.skillsDir), scanLastRuns(dirs)]);
    const body: SkillsResponse = {
      skills: skills.map((s): SkillSummary => ({ name: s.name, description: s.description, lastRun: runs.get(s.name) ?? null, confirmRequired: s.confirmRequired, art: s.art })),
    };
    return c.json(body);
  });

  /** Starts a worker that reads the skill's tracked SKILL.md; the server picks the agent name so the form rules never reach the user. */
  api.post("/skills/:name/run", limit, async (c) => {
    const input = await readObject(c);
    if (!input) return badRequest(c, "body must be a JSON object");
    if (input.args !== undefined && typeof input.args !== "string") return badRequest(c, "args must be a string");
    const args = (input.args ?? "").toString().trim();
    if (args.length > ARGS_MAX) return badRequest(c, `args must be at most ${ARGS_MAX} characters`);
    const skill = (await readSkills(d.config.skillsDir)).find((s) => s.name === c.req.param("name"));
    if (!skill) return c.json({ error: "Skill not found." }, 404);
    const task = taskFor(skill, args);
    for (let attempt = 1; ; attempt++) {
      const name = `sk-${slugOf(skill.name)}-${randomBytes(SUFFIX_BYTES).toString("hex")}`;
      try {
        return c.json(await d.agents.spawn({ name, task }), 201);
      } catch (err) {
        if (err instanceof AgentError && err.code === "exists" && attempt < NAME_TRIES) continue;
        return failAgent(c, err);
      }
    }
  });
}
