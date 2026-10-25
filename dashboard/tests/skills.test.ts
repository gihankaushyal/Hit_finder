import { describe, it, expect, afterEach } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { scanLastRuns, withWorktreeDirs } from "../server/skills/lastRun";
import { makeAgentRig, AUTH, type AgentRig } from "./helpers/agentRig";
import type { SkillsResponse } from "../shared/types";

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
  const setup = async () => {
    rig = await makeAgentRig({ DASH_SKILLS_DIR: path.join(FIX, "skills"), DASH_TRANSCRIPTS_DIR: TRANSCRIPTS });
  };
  afterEach(async () => rig?.cleanup());
  const list = async () => ((await (await rig.app.request("/api/skills", { headers: AUTH })).json()) as SkillsResponse).skills;

  it("lists skills with last run and the art from the SKILL.md header", async () => {
    await setup();
    const res = await rig.app.request("/api/skills", { headers: AUTH });
    expect(res.status).toBe(200);
    const skills = ((await res.json()) as SkillsResponse).skills;
    expect(skills).toHaveLength(4);
    expect(skills.find((s) => s.name === "alpha")).toEqual({ name: "alpha", description: "Alpha does a thing across two lines", lastRun: "2026-10-05T12:00:00.000Z", art: "docs" });
    expect(skills.find((s) => s.name === "slurm-job")).toEqual({ name: "slurm-job", description: "Submit a job: needs a config", lastRun: "2026-09-01T00:00:00.000Z", art: null });
  });

  it("lists a skill whose header cannot be read", async () => {
    await setup();
    expect((await list()).find((s) => s.name === "broken")).toMatchObject({ art: null, lastRun: null });
  });

  it("has no run endpoint: skills run from Claude Code, not the dashboard", async () => {
    await setup();
    const res = await rig.app.request("/api/skills/alpha/run", { method: "POST", headers: AUTH, body: JSON.stringify({}) });
    expect(res.status).toBe(404);
  });

  it("lists a skill name once when two folders share it", async () => {
    const skillsDir = tmp();
    for (const dir of ["one", "two"]) {
      fs.mkdirSync(path.join(skillsDir, dir));
      fs.writeFileSync(path.join(skillsDir, dir, "SKILL.md"), `---\nname: same\ndescription: from ${dir}\n---\n`);
    }
    rig = await makeAgentRig({ DASH_SKILLS_DIR: skillsDir, DASH_TRANSCRIPTS_DIR: TRANSCRIPTS });
    expect((await list()).map((s) => [s.name, s.description])).toEqual([["same", "from one"]]);
  });
});
