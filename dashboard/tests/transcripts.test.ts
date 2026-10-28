import { describe, it, expect, beforeEach, afterEach } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { createTranscriptProbe } from "../server/agents/transcripts";

let configDir: string;
beforeEach(() => {
  configDir = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), "dash-transcripts-")));
});
afterEach(() => fs.rmSync(configDir, { recursive: true, force: true }));

describe("transcript probe", () => {
  it("finds a saved session in any project folder", async () => {
    fs.mkdirSync(path.join(configDir, "projects", "-some-project"), { recursive: true });
    fs.writeFileSync(path.join(configDir, "projects", "-some-project", "abc-123.jsonl"), "{}\n");
    const probe = createTranscriptProbe(configDir);
    expect(await probe("abc-123")).toBe(true);
    expect(await probe("zzz-999")).toBe(false);
  });

  it("assumes the session is there when the projects folder cannot be read", async () => {
    expect(await createTranscriptProbe(path.join(configDir, "nope"))("abc-123")).toBe(true);
  });

  it("assumes the session is there for an id that is not safe to look up", async () => {
    fs.mkdirSync(path.join(configDir, "projects"), { recursive: true });
    expect(await createTranscriptProbe(configDir)("../../etc/passwd")).toBe(true);
  });

  it("looks under HOME, where the chief's own claude writes, and ignores the dashboard's CLAUDE_CONFIG_DIR", async () => {
    const home = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), "dash-home-")));
    fs.mkdirSync(path.join(home, ".claude", "projects", "p"), { recursive: true });
    fs.writeFileSync(path.join(home, ".claude", "projects", "p", "s-1.jsonl"), "{}\n");
    const saved = { HOME: process.env.HOME, CLAUDE_CONFIG_DIR: process.env.CLAUDE_CONFIG_DIR };
    process.env.HOME = home;
    process.env.CLAUDE_CONFIG_DIR = configDir; // somewhere else, empty
    try {
      expect(await createTranscriptProbe()("s-1")).toBe(true);
    } finally {
      for (const [k, v] of Object.entries(saved)) v === undefined ? delete process.env[k] : (process.env[k] = v);
      fs.rmSync(home, { recursive: true, force: true });
    }
  });
});
