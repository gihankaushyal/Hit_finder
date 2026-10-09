import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { chromium, defineConfig } from "@playwright/test";

// Use Playwright's own browser when it is installed. Otherwise fall back to any Chromium already in the
// Playwright cache (never downloaded here), or to E2E_CHROMIUM_PATH if set.
function chromiumPath(): string | undefined {
  if (process.env.E2E_CHROMIUM_PATH) return process.env.E2E_CHROMIUM_PATH;
  if (fs.existsSync(chromium.executablePath())) return undefined;
  const cache = path.join(os.homedir(), ".cache", "ms-playwright");
  try {
    for (const d of fs.readdirSync(cache).filter((n) => /^chromium-\d+$/.test(n)).sort().reverse()) {
      const exe = path.join(cache, d, "chrome-linux64", "chrome");
      if (fs.existsSync(exe)) return exe;
    }
  } catch {
    // no cache directory
  }
  return undefined;
}

export default defineConfig({
  testDir: "e2e",
  testMatch: "*.spec.ts",
  globalSetup: "./e2e/global-setup.ts",
  workers: 1,
  fullyParallel: false,
  timeout: 60_000,
  expect: { timeout: 10_000 },
  reporter: [["list"]],
  use: {
    viewport: { width: 1440, height: 900 },
    colorScheme: "dark",
    trace: "retain-on-failure",
    launchOptions: { executablePath: chromiumPath() },
  },
  projects: [{ name: "chromium", use: { browserName: "chromium" } }],
});
