import { spawnSync } from "node:child_process";
import path from "node:path";
import { DASHBOARD_DIR } from "./harness";

/** The server serves dashboard/dist, so build it once before any spec runs. */
export default function globalSetup(): void {
  const vite = path.join(DASHBOARD_DIR, "node_modules", ".bin", "vite");
  const r = spawnSync(vite, ["build"], { cwd: DASHBOARD_DIR, stdio: "inherit" });
  if (r.status !== 0) throw new Error("e2e: vite build failed");
}
