import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import type { MiddlewareHandler } from "hono";
import { getCookie } from "hono/cookie";

export const TOKEN_COOKIE = "dash_token";
const TOKEN_BYTES = 32;
const DIR_MODE = 0o700;
const FILE_MODE = 0o600;

const TOKEN_PATTERN = new RegExp(`^[0-9a-f]{${TOKEN_BYTES * 2}}$`);

type TokenRead = { kind: "valid"; token: string } | { kind: "malformed" } | { kind: "missing" };

function readToken(file: string): TokenRead {
  try {
    const existing = fs.readFileSync(file, "utf8").trim();
    if (!TOKEN_PATTERN.test(existing)) return { kind: "malformed" };
    fs.chmodSync(file, FILE_MODE);
    return { kind: "valid", token: existing };
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code !== "ENOENT") throw err;
    return { kind: "missing" };
  }
}

export function loadOrCreateToken(stateDir: string): string {
  const file = path.join(stateDir, "token");
  fs.mkdirSync(stateDir, { recursive: true, mode: DIR_MODE });
  fs.chmodSync(stateDir, DIR_MODE);
  const first = readToken(file);
  if (first.kind === "valid") return first.token;
  // Only a file we read and found malformed is removed; never one that merely appeared since.
  if (first.kind === "malformed") fs.rmSync(file, { force: true });
  const token = crypto.randomBytes(TOKEN_BYTES).toString("hex");
  try {
    // "wx" makes concurrent starters agree on a single winner.
    fs.writeFileSync(file, token, { mode: FILE_MODE, flag: "wx" });
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code !== "EEXIST") throw err;
    const winner = readToken(file);
    if (winner.kind === "valid") return winner.token;
    throw err;
  }
  fs.chmodSync(file, FILE_MODE);
  return token;
}

/** Constant-time comparison that never throws on length mismatch. */
export function tokensMatch(expected: string, given: string | undefined | null): boolean {
  if (!given) return false;
  const a = Buffer.from(expected);
  const b = Buffer.from(given);
  if (a.length !== b.length) return false;
  return crypto.timingSafeEqual(a, b);
}

function hasValidBearer(c: Parameters<MiddlewareHandler>[0], token: string): boolean {
  const auth = c.req.header("Authorization");
  return !!auth?.startsWith("Bearer ") && tokensMatch(token, auth.slice("Bearer ".length));
}

export function requestHasValidToken(c: Parameters<MiddlewareHandler>[0], token: string): boolean {
  return hasValidBearer(c, token) || tokensMatch(token, getCookie(c, TOKEN_COOKIE));
}

const SAFE_METHODS = new Set(["GET", "HEAD"]);
const JSON_MEDIA_TYPE = "application/json";
const SAME_ORIGIN_FETCH_SITES = new Set(["same-origin", "none"]);

/**
 * Cookies are not scoped by port and SameSite treats every 127.0.0.1 port as same-site, so another
 * local service could forge state-changing requests. Mutations must therefore be JSON (which a
 * cross-origin page cannot send without a preflight we never grant) and, when the browser says where
 * they came from, come from this very origin. A Bearer token cannot be attached by a foreign page.
 */
function csrfRejection(c: Parameters<MiddlewareHandler>[0], viaBearer: boolean): Response | null {
  if (SAFE_METHODS.has(c.req.method)) return null;
  const origin = c.req.header("Origin");
  if (!viaBearer && origin !== undefined && origin !== new URL(c.req.url).origin) {
    return c.json({ error: "cross-origin request refused" }, 403);
  }
  const site = c.req.header("Sec-Fetch-Site");
  if (site !== undefined && !SAME_ORIGIN_FETCH_SITES.has(site)) {
    return c.json({ error: "cross-site request refused" }, 403);
  }
  const mediaType = (c.req.header("Content-Type") ?? "").split(";")[0].trim().toLowerCase();
  if (mediaType !== JSON_MEDIA_TYPE) return c.json({ error: "Content-Type must be application/json" }, 415);
  return null;
}

export function authMiddleware(token: string): MiddlewareHandler {
  return async (c, next) => {
    if (!requestHasValidToken(c, token)) return c.json({ error: "unauthorized" }, 401);
    const rejected = csrfRejection(c, hasValidBearer(c, token));
    if (rejected) return rejected;
    await next();
  };
}
