import { it, expect } from "vitest";
import { execRunner } from "../server/runner";
it("captures stdout and exit code without throwing", async () => {
  const ok = await execRunner(process.execPath, ["-e", "console.log('hi')"]);
  expect(ok).toMatchObject({ stdout: "hi\n", code: 0 });
  const bad = await execRunner(process.execPath, ["-e", "console.error('no'); process.exit(3)"]);
  expect(bad.code).toBe(3);
  expect(bad.stderr).toContain("no");
});
it("reports a missing binary as code 127", async () => {
  const r = await execRunner("/nonexistent/bin", []);
  expect(r.code).toBe(127);
});
