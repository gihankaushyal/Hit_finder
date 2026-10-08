import fs from "node:fs";
import path from "node:path";
import { Hono } from "hono";
import { setCookie } from "hono/cookie";
import type { Config } from "./config";
import type { EventBus } from "./events";
import { TOKEN_COOKIE, authMiddleware, requestHasValidToken, tokensMatch } from "./auth";

export interface Deps {
  config: Config;
  token: string;
  bus: EventBus;
  routes?: (api: Hono) => void;
}

const CONTENT_TYPES: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".ico": "image/x-icon",
  ".woff": "font/woff",
  ".woff2": "font/woff2",
  ".map": "application/json; charset=utf-8",
};

function readFileIfFile(p: string): Buffer | null {
  try {
    return fs.statSync(p).isFile() ? fs.readFileSync(p) : null;
  } catch {
    return null;
  }
}

export function createApp(deps: Deps): Hono {
  const { config, token } = deps;
  const app = new Hono();

  const api = new Hono();
  api.use("*", authMiddleware(token));
  api.get("/health", (c) => c.json({ ok: true }));
  deps.routes?.(api);
  app.route("/api", api);

  app.get("*", (c) => {
    const given = c.req.query("token");
    if (given !== undefined) {
      if (!tokensMatch(token, given)) return c.text("Invalid token.", 401);
      setCookie(c, TOKEN_COOKIE, token, { httpOnly: true, sameSite: "Strict", path: "/" });
      return c.redirect("/", 302);
    }
    if (!requestHasValidToken(c, token)) {
      return c.text("Unauthorized. Open the tokenised URL printed by the server.", 401);
    }
    const root = path.resolve(config.webDist);
    if (!fs.existsSync(root)) return c.json({ error: "dashboard build not found (run npm run build)" }, 404);

    const requested = path.resolve(root, "." + decodeURIComponent(c.req.path));
    const inRoot = requested === root || requested.startsWith(root + path.sep);
    const direct = inRoot ? readFileIfFile(requested) : null;
    const body = direct ?? readFileIfFile(path.join(root, "index.html"));
    if (!body) return c.json({ error: "index.html not found in dist" }, 404);
    const ext = direct ? path.extname(requested) : ".html";
    return c.body(new Uint8Array(body), 200, {
      "Content-Type": CONTENT_TYPES[ext] ?? "application/octet-stream",
    });
  });

  return app;
}
