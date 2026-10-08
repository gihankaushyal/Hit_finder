import type { Hono } from "hono";
import { createApp } from "./app";
import type { Config } from "./config";
import type { EventBus } from "./events";
import type { Gh } from "./lib/gh";
import type { KanbanService } from "./lib/kanbanService";
import type { TestRunner } from "./lib/testRunner";
import { ciRoutes } from "./routes/ci";
import { eventRoutes } from "./routes/events";
import { issueRoutes } from "./routes/issues";
import { kanbanRoutes } from "./routes/kanban";
import { prRoutes } from "./routes/prs";
import { testRoutes } from "./routes/tests";

export interface ServerDeps {
  config: Config;
  token: string;
  bus: EventBus;
  gh: Gh;
  tests: TestRunner;
  kanban: KanbanService;
}

/** Builds the app with every route module mounted behind token auth (no listening, no side effects). */
export function buildServer(d: ServerDeps): Hono {
  return createApp({
    config: d.config,
    token: d.token,
    bus: d.bus,
    routes(api) {
      prRoutes(api, { gh: d.gh });
      issueRoutes(api, { gh: d.gh, bus: d.bus });
      ciRoutes(api, { gh: d.gh });
      testRoutes(api, { tests: d.tests });
      eventRoutes(api, { bus: d.bus });
      kanbanRoutes(api, { service: d.kanban });
    },
  });
}
