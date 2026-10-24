import { describe, it, expect, afterEach } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { scanLastRuns, withWorktreeDirs } from "../server/skills/lastRun";
import { makeAgentRig, AUTH, type AgentRig } from "./helpers/agentRig";
import type { AgentSummary, AgentsResponse, SkillsResponse } from "../shared/types";

const FIX = path.join(__dirname, "fixtures");
const TRANSCRIPTS = path.join(FIX, "transcripts");
const tmpDirs: string[] = [];
const tmp = () => {
  const d = fs.mkdtempSync(path.join(os.tmpdir(), "dash-skills-"));
  tmpDirs.push(d);
  return d;
};
afterEach(() => {
  for (const d of tmpDirs.splice(0)) fs.rmSync(d, { recursive: true, force: true });
});

const slash = (name: string, at: string) => JSON.stringify({ type: "user", timestamp: at, message: { role: "user", content: `<command-name>/${name}</command-name>` } }) + "\n";

describe("scanLastRuns", () => {
  it("keeps the newest time per skill, counting only real invocations", async () => {
    const runs = await scanLastRuns(TRANSCRIPTS);
    expect(runs.get("alpha")).toBe("2026-10-05T12:00:00.000Z");
    // slurm-job: the user only quoted the tag on 10-06 and the 10-07 call errored, so only b.jsonl counts
    expect(runs.get("slurm-job")).toBe("2026-09-01T00:00:00.000Z");
    expect(runs.has("Bash")).toBe(false);
  });
  it("gives an empty map for a missing directory", async () => {
    expect((await scanLastRuns(path.join(FIX, "nope"))).size).toBe(0);
  });
  it("reads only what was appended, and waits for a half-written line", async () => {
    const dir = tmp();
    const file = path.join(dir, "s.jsonl");
    fs.writeFileSync(file, slash("alpha", "2026-10-01T00:00:00.000Z"));
    expect((await scanLastRuns(dir)).get("alpha")).toBe("2026-10-01T00:00:00.000Z");
    const next = slash("alpha", "2026-10-02T00:00:00.000Z");
    fs.appendFileSync(file, next.slice(0, 30));
    expect((await scanLastRuns(dir)).get("alpha")).toBe("2026-10-01T00:00:00.000Z");
    fs.appendFileSync(file, next.slice(30));
    expect((await scanLastRuns(dir)).get("alpha")).toBe("2026-10-02T00:00:00.000Z");
  });
  it("starts over when a transcript is replaced by a shorter one", async () => {
    const dir = tmp();
    const file = path.join(dir, "s.jsonl");
    fs.writeFileSync(file, slash("alpha", "2026-10-01T00:00:00.000Z") + slash("beta", "2026-10-01T00:00:00.000Z"));
    await scanLastRuns(dir);
    fs.writeFileSync(file, slash("beta", "2026-10-03T00:00:00.000Z"));
    const runs = await scanLastRuns(dir);
    expect(runs.has("alpha")).toBe(false);
    expect(runs.get("beta")).toBe("2026-10-03T00:00:00.000Z");
  });
  it("shares one scan between overlapping requests", () => {
    const dir = tmp();
    expect(scanLastRuns(dir)).toBe(scanLastRuns(dir));
  });
});

describe("withWorktreeDirs", () => {
  it("adds the transcript folders of the repo's worktrees and nothing else", async () => {
    const parent = tmp();
    for (const n of ["-x-Repo", "-x-Repo--claude-worktrees-a", "-x-Repo--claude-worktrees-b", "-x-Repo-other", "-x-Other"]) fs.mkdirSync(path.join(parent, n));
    const dirs = await withWorktreeDirs(path.join(parent, "-x-Repo"));
    expect(dirs.map((d) => path.basename(d))).toEqual(["-x-Repo", "-x-Repo--claude-worktrees-a", "-x-Repo--claude-worktrees-b"]);
  });
  it("falls back to the main folder when the parent is unreadable", async () => {
    expect(await withWorktreeDirs("/nonexistent-dir-for-test/x")).toEqual(["/nonexistent-dir-for-test/x"]);
  });
});

describe("skills routes", () => {
  let rig: AgentRig;
  const setup = async () => {
    rig = await makeAgentRig({ DASH_SKILLS_DIR: path.join(FIX, "skills"), DASH_TRANSCRIPTS_DIR: TRANSCRIPTS });
  };
  afterEach(async () => rig?.cleanup());
  const post = (name: string, body: unknown) => rig.app.request(`/api/skills/${encodeURIComponent(name)}/run`, { method: "POST", headers: AUTH, body: JSON.stringify(body) });

  it("lists skills with last run, the confirm flag and the art from the SKILL.md header", async () => {
    await setup();
    const res = await rig.app.request("/api/skills", { headers: AUTH });
    expect(res.status).toBe(200);
    const skills = ((await res.json()) as SkillsResponse).skills;
    expect(skills).toHaveLength(4);
    expect(skills.find((s) => s.name === "alpha")).toEqual({ name: "alpha", description: "Alpha does a thing across two lines", lastRun: "2026-10-05T12:00:00.000Z", confirmRequired: false, art: "docs" });
    expect(skills.find((s) => s.name === "slurm-job")).toEqual({ name: "slurm-job", description: "Submit a job: needs a config", lastRun: "2026-09-01T00:00:00.000Z", confirmRequired: true, art: null });
  });

  it("lists a skill whose header cannot be read, and makes it ask for confirmation", async () => {
    await setup();
    const skills = ((await (await rig.app.request("/api/skills", { headers: AUTH })).json()) as SkillsResponse).skills;
    expect(skills.find((s) => s.name === "broken")).toMatchObject({ confirmRequired: true, art: null, lastRun: null });
  });

  it("starts an agent that reads the skill's SKILL.md, with a server-chosen valid name", async () => {
    await setup();
    const res = await post("alpha", { args: " configs/x.yaml " });
    expect(res.status).toBe(201);
    const agent = (await res.json()) as AgentSummary;
    expect(agent.name).toMatch(/^sk-alpha-[0-9a-f]{4}$/);
    expect(agent.task).toContain("Read .claude/skills/alpha/SKILL.md");
    expect(agent.task).toContain("Arguments: configs/x.yaml");
    const list = (await (await rig.app.request("/api/agents", { headers: AUTH })).json()) as AgentsResponse;
    expect(list.agents.map((a) => a.name)).toContain(agent.name);
  });

  it("points at the skill's folder, and shortens a long skill name to fit the agent name rules", async () => {
    await setup();
    const slurm = (await (await post("slurm-job", {})).json()) as AgentSummary;
    expect(slurm.task).toContain(".claude/skills/beta/SKILL.md");
    expect(slurm.task).not.toContain("Arguments:");
    const long = (await (await post("A_Very_Long_Skill_Name_Indeed", {})).json()) as AgentSummary;
    expect(long.name).toMatch(/^sk-a-very-long-sk-[0-9a-f]{4}$/);
  });

  it("rejects an unknown skill and bad arguments", async () => {
    await setup();
    expect((await post("nope", {})).status).toBe(404);
    expect((await post("alpha", { args: 5 })).status).toBe(400);
    expect((await post("alpha", { args: "x".repeat(501) })).status).toBe(400);
  });
});
