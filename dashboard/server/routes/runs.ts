import type { Hono } from "hono";
import { respondWithError } from "../lib/routeErrors";
import type { RunScanner } from "../lib/runScan";

export function runRoutes(api: Hono, d: { scanner: RunScanner }): void {
  api.get("/runs", async (c) => {
    try {
      return c.json(await d.scanner.scan());
    } catch (err) {
      return respondWithError(c, err);
    }
  });
}
