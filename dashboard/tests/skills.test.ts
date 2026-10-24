import { describe, it, expect, afterEach } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { execFileSync } from "node:child_process";
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
  it("keeps the same answer when scans over different folder lists overlap", async () => {
    const a = tmp();
    const b = tmp();
    fs.writeFileSync(path.join(a, "s.jsonl"), slash("alpha", "2026-10-01T00:00:00.000Z") + slash("alpha", "2026-10-02T00:00:00.000Z"));
    fs.writeFileSync(path.join(b, "t.jsonl"), slash("beta", "2026-10-03T00:00:00.000Z"));
    const [first, second] = await Promise.all([scanLastRuns(a), scanLastRuns([a, b])]);
    expect(first.get("alpha")).toBe("2026-10-02T00:00:00.000Z");
    expect(second.get("alpha")).toBe("2026-10-02T00:00:00.000Z");
    expect(second.get("beta")).toBe("2026-10-03T00:00:00.000Z");
  });
  it("shares one scan between overlapping requests", () => {
    const dir = tmp();
    expect(scanLastRuns(dir)).toBe(scanLastRuns(dir));
  });
});

describe("skill names in transcripts", () => {
  it("counts /name and prefix:name calls under the bare skill name", async () => {
    const dir = tmp();
    const call = (id: string, skill: string, at: string) => JSON.stringify({ type: "assistant", timestamp: at, message: { role: "assistant", content: [{ type: "tool_use", id, name: "Skill", input: { skill } }] } }) + "\n";
    const ok = (id: string, at: string) => JSON.stringify({ type: "user", timestamp: at, message: { role: "user", content: [{ type: "tool_result", tool_use_id: id }] } }) + "\n";
    fs.writeFileSync(path.join(dir, "a.jsonl"), call("t1", "/slurm-job", "2026-10-01T00:00:00Z") + ok("t1", "2026-10-01T00:00:01Z") + call("t2", "project:docs-refresh", "2026-10-02T00:00:00Z") + ok("t2", "2026-10-02T00:00:01Z"));
    const runs = await scanLastRuns([dir]);
    expect(runs.get("slurm-job")).toBe("2026-10-01T00:00:00Z");
    expect(runs.get("docs-refresh")).toBe("2026-10-02T00:00:00Z");
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
  const git = (...a: string[]) => execFileSync("git", ["-C", rig.repo, ...a], { encoding: "utf8" });
  /** The skills live in the rig's repo and are committed on its base branch, as a merged skill would be. */
  const setup = async () => {
    rig = await makeAgentRig({ DASH_TRANSCRIPTS_DIR: TRANSCRIPTS });
    fs.cpSync(path.join(FIX, "skills"), rig.cfg.skillsDir, { recursive: true });
    git("add", ".claude/skills");
    git("commit", "-q", "-m", "skills");
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
    expect(skills.find((s) => s.name === "A_Very_Long_Skill_Name_Indeed")?.confirmRequired).toBe(true); // no header key: ask by default
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
    const slurm = (await (await post("slurm-job", { confirmed: true })).json()) as AgentSummary;
    expect(slurm.task).toContain(".claude/skills/beta/SKILL.md");
    expect(slurm.task).not.toContain("Arguments:");
    const long = (await (await post("A_Very_Long_Skill_Name_Indeed", { confirmed: true })).json()) as AgentSummary;
    expect(long.name).toMatch(/^sk-a-very-long-sk-[0-9a-f]{4}$/);
  });

  it("refuses to run a skill that the agent base does not have, or has in a different version", async () => {
    await setup();
    fs.appendFileSync(path.join(rig.cfg.skillsDir, "alpha", "SKILL.md"), "\nlocal edit\n");
    const differs = await post("alpha", {});
    expect(differs.status).toBe(409);
    expect(((await differs.json()) as { error: string }).error).toContain("differs");
    fs.mkdirSync(path.join(rig.cfg.skillsDir, "fresh"));
    fs.writeFileSync(path.join(rig.cfg.skillsDir, "fresh", "SKILL.md"), "---\nname: fresh\ndashboard-confirm: false\n---\nbody\n");
    const missing = await post("fresh", {});
    expect(missing.status).toBe(409);
    expect(((await missing.json()) as { error: string }).error).toContain("not on");
    expect(((await (await rig.app.request("/api/agents", { headers: AUTH })).json()) as AgentsResponse).agents).toHaveLength(0);
  });

  it("refuses a skill that needs confirmation unless the request says it was confirmed", async () => {
    await setup();
    for (const body of [{}, { confirmed: false }, { confirmed: "true" }]) expect((await post("slurm-job", body)).status).toBe(400);
    expect(((await (await rig.app.request("/api/agents", { headers: AUTH })).json()) as AgentsResponse).agents).toHaveLength(0);
    expect((await post("slurm-job", { confirmed: true })).status).toBe(201);
    expect((await post("alpha", {})).status).toBe(201); // opted out with dashboard-confirm: false
  });

  it("counts a run started from the dashboard as the skill's last run", async () => {
    await setup();
    const before = Date.now();
    await post("alpha", {});
    const skills = ((await (await rig.app.request("/api/skills", { headers: AUTH })).json()) as SkillsResponse).skills;
    expect(Date.parse(skills.find((s) => s.name === "alpha")!.lastRun!)).toBeGreaterThanOrEqual(before - 1000);
    expect(skills.find((s) => s.name === "slurm-job")?.lastRun).toBe("2026-09-01T00:00:00.000Z"); // untouched
  });

  it("lists a skill name once when two folders share it", async () => {
    const skillsDir = tmp();
    for (const dir of ["one", "two"]) {
      fs.mkdirSync(path.join(skillsDir, dir));
      fs.writeFileSync(path.join(skillsDir, dir, "SKILL.md"), `---\nname: same\ndescription: from ${dir}\n---\n`);
    }
    rig = await makeAgentRig({ DASH_SKILLS_DIR: skillsDir, DASH_TRANSCRIPTS_DIR: TRANSCRIPTS });
    const skills = ((await (await rig.app.request("/api/skills", { headers: AUTH })).json()) as SkillsResponse).skills;
    expect(skills.map((s) => [s.name, s.description])).toEqual([["same", "from one"]]);
  });

  it("rejects an unknown skill and bad arguments", async () => {
    await setup();
    expect((await post("nope", {})).status).toBe(404);
    expect((await post("alpha", { args: 5 })).status).toBe(400);
    expect((await post("alpha", { args: "x".repeat(501) })).status).toBe(400);
  });
});
