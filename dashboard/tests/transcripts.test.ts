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
});
