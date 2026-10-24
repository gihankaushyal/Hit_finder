import { describe, it, expect } from "vitest";
import path from "node:path";
import { scanLastRuns } from "../server/skills/lastRun";
import { makeAgentRig, AUTH } from "./helpers/agentRig";
import type { SkillsResponse } from "../shared/types";

const FIX = path.join(__dirname, "fixtures");
const TRANSCRIPTS = path.join(FIX, "transcripts");

describe("scanLastRuns", () => {
  it("keeps the newest time per skill across tool calls and slash commands", async () => {
    const runs = await scanLastRuns(TRANSCRIPTS);
    expect(runs.get("alpha")).toBe("2026-10-05T12:00:00.000Z");
    expect(runs.get("slurm-job")).toBe("2026-09-01T00:00:00.000Z");
    expect(runs.has("Bash")).toBe(false);
  });
  it("gives an empty map for a missing directory", async () => {
    expect((await scanLastRuns(path.join(FIX, "nope"))).size).toBe(0);
  });
});

describe("GET /api/skills", () => {
  it("lists skills with last run and the confirm flag", async () => {
    const rig = await makeAgentRig({ DASH_SKILLS_DIR: path.join(FIX, "skills"), DASH_TRANSCRIPTS_DIR: TRANSCRIPTS });
    try {
      const res = await rig.app.request("/api/skills", { headers: AUTH });
      expect(res.status).toBe(200);
      expect(((await res.json()) as SkillsResponse).skills).toEqual([
        { name: "alpha", description: "Alpha does a thing", lastRun: "2026-10-05T12:00:00.000Z", confirmRequired: false },
        { name: "slurm-job", description: "Submit a job", lastRun: "2026-09-01T00:00:00.000Z", confirmRequired: true },
      ]);
    } finally {
      await rig.cleanup();
    }
  });
});
