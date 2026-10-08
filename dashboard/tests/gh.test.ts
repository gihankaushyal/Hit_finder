import { describe, it, expect } from "vitest";
import { createGh, GhError } from "../server/lib/gh";
import type { Runner, RunResult } from "../server/runner";

const cfg = { ghBin: "gh", repoRoot: "/repo" };

function fake(result: RunResult) {
  const calls: { cmd: string; args: string[]; opts?: { cwd?: string } }[] = [];
  const runner: Runner = async (cmd, args, opts) => {
    calls.push({ cmd, args, opts });
    return result;
  };
  return { runner, calls };
}

describe("createGh", () => {
  it("runs gh with an args array and cwd, and parses JSON", async () => {
    const { runner, calls } = fake({ stdout: '[{"number":1}]', stderr: "", code: 0 });
    const out = await createGh(runner, cfg).json<{ number: number }[]>(["pr", "list"]);
    expect(out).toEqual([{ number: 1 }]);
    expect(calls).toEqual([{ cmd: "gh", args: ["pr", "list"], opts: { cwd: "/repo" } }]);
  });
  it("uses cfg.ghBin as the command", async () => {
    const { runner, calls } = fake({ stdout: "x", stderr: "", code: 0 });
    await createGh(runner, { ghBin: "/x/gh", repoRoot: "/repo" }).text(["a"]);
    expect(calls[0].cmd).toBe("/x/gh");
  });
  it("throws GhError with stderr and code on non-zero exit", async () => {
    const { runner } = fake({ stdout: "", stderr: "boom", code: 4 });
    const err = (await createGh(runner, cfg).json(["pr", "list"]).catch((e: unknown) => e)) as GhError;
    expect(err).toBeInstanceOf(GhError);
    expect(err.message).toContain("boom");
    expect(err.code).toBe(4);
    expect(err.args).toEqual(["pr", "list"]);
  });
  it("throws GhError on invalid JSON", async () => {
    const { runner } = fake({ stdout: "not json", stderr: "", code: 0 });
    const err = (await createGh(runner, cfg).json(["x"]).catch((e: unknown) => e)) as GhError;
    expect(err).toBeInstanceOf(GhError);
    expect(err.message).toContain("invalid JSON");
    expect(err.code).toBe(-1);
    expect(err.publicMessage).toBe("GitHub CLI returned invalid JSON");
  });
  it("exposes a fixed publicMessage and truncates stderr in message", async () => {
    const { runner } = fake({ stdout: "", stderr: "x".repeat(2000), code: 2 });
    const err = (await createGh(runner, cfg).text(["a"]).catch((e: unknown) => e)) as GhError;
    expect(err.publicMessage).toBe("GitHub CLI failed");
    expect(err.message.length).toBeLessThan(700);
  });
  it("text trims stdout and throws on failure", async () => {
    const ok = fake({ stdout: "  hello\n", stderr: "", code: 0 });
    expect(await createGh(ok.runner, cfg).text(["a"])).toBe("hello");
    const bad = fake({ stdout: "", stderr: "nope", code: 1 });
    await expect(createGh(bad.runner, cfg).text(["a"])).rejects.toBeInstanceOf(GhError);
  });
});
