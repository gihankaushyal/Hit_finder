import type { Hono } from "hono";
import { respondWithError } from "../lib/routeErrors";
import type { RunScanner } from "../lib/runScan";

export function runRoutes(api: Hono, d: { scanner: RunScanner }): void {
  api.get("/runs", (c) => {
    try {
      return c.json(d.scanner.scan());
    } catch (err) {
      return respondWithError(c, err);
    }
  });
}
