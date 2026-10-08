import type { Hono } from "hono";
import type { TestRunner } from "../lib/testRunner";

const ALREADY_RUNNING = "A test run is already in progress";

export function testRoutes(api: Hono, d: { tests: TestRunner }): void {
  api.get("/tests", (c) => c.json(d.tests.state()));
  api.post("/tests/run", (c) => {
    if (!d.tests.start()) return c.json({ error: ALREADY_RUNNING }, 409);
    return c.json({ started: true }, 202);
  });
}
