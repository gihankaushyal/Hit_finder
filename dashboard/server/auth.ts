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

function readValidToken(file: string): string | null {
  try {
    const existing = fs.readFileSync(file, "utf8").trim();
    if (!TOKEN_PATTERN.test(existing)) return null;
    fs.chmodSync(file, FILE_MODE);
    return existing;
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code !== "ENOENT") throw err;
    return null;
  }
}

export function loadOrCreateToken(stateDir: string): string {
  const file = path.join(stateDir, "token");
  fs.mkdirSync(stateDir, { recursive: true, mode: DIR_MODE });
  fs.chmodSync(stateDir, DIR_MODE);
  const existing = readValidToken(file);
  if (existing) return existing;
  const token = crypto.randomBytes(TOKEN_BYTES).toString("hex");
  try {
    // Replace a malformed file; "wx" below then makes concurrent starters agree.
    fs.rmSync(file, { force: true });
    fs.writeFileSync(file, token, { mode: FILE_MODE, flag: "wx" });
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code !== "EEXIST") throw err;
    const winner = readValidToken(file);
    if (winner) return winner;
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

export function requestHasValidToken(c: Parameters<MiddlewareHandler>[0], token: string): boolean {
  const auth = c.req.header("Authorization");
  if (auth?.startsWith("Bearer ") && tokensMatch(token, auth.slice("Bearer ".length))) return true;
  return tokensMatch(token, getCookie(c, TOKEN_COOKIE));
}

export function authMiddleware(token: string): MiddlewareHandler {
  return async (c, next) => {
    if (!requestHasValidToken(c, token)) return c.json({ error: "unauthorized" }, 401);
    await next();
  };
}
