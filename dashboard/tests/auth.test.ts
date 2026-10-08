import { describe, it, expect, beforeEach } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { loadOrCreateToken } from "../server/auth";
import { EventBus } from "../server/events";
import { createApp } from "../server/app";
import { loadConfig } from "../server/config";

let stateDir: string;
let token: string;
let app: ReturnType<typeof createApp>;

beforeEach(() => {
  stateDir = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "dash-")), "state");
  token = loadOrCreateToken(stateDir);
  const config = { ...loadConfig({}), stateDir, webDist: path.join(stateDir, "no-dist") };
  app = createApp({
    config,
    token,
    bus: new EventBus(),
    routes: (api) => {
      api.get("/secret", (c) => c.json({ secret: true }));
    },
  });
});

describe("loadOrCreateToken", () => {
  it("creates a 0600 token in a 0700 dir and reuses it", () => {
    const file = path.join(stateDir, "token");
    expect(token).toMatch(/^[0-9a-f]{64}$/);
    expect(fs.statSync(file).mode & 0o777).toBe(0o600);
    expect(fs.statSync(stateDir).mode & 0o777).toBe(0o700);
    expect(loadOrCreateToken(stateDir)).toBe(token);
  });
});

describe("auth", () => {
  it("rejects /api/health without a token", async () => {
    expect((await app.request("/api/health")).status).toBe(401);
  });
  it("rejects wrong tokens of equal and different length without throwing", async () => {
    const same = "0".repeat(token.length);
    expect((await app.request("/api/health", { headers: { Authorization: `Bearer ${same}` } })).status).toBe(401);
    expect((await app.request("/api/health", { headers: { Authorization: "Bearer short" } })).status).toBe(401);
  });
  it("accepts Bearer and cookie", async () => {
    const b = await app.request("/api/health", { headers: { Authorization: `Bearer ${token}` } });
    expect(b.status).toBe(200);
    expect(await b.json()).toEqual({ ok: true });
    const c = await app.request("/api/health", { headers: { Cookie: `dash_token=${token}` } });
    expect(c.status).toBe(200);
  });
  it("protects routes registered via deps.routes", async () => {
    expect((await app.request("/api/secret")).status).toBe(401);
    const ok = await app.request("/api/secret", { headers: { Authorization: `Bearer ${token}` } });
    expect(await ok.json()).toEqual({ secret: true });
  });
  it("sets a hardened cookie and redirects for a good ?token", async () => {
    const r = await app.request(`/?token=${token}`);
    expect(r.status).toBe(302);
    expect(r.headers.get("location")).toBe("/");
    const cookie = r.headers.get("set-cookie") ?? "";
    expect(cookie).toContain(`dash_token=${token}`);
    expect(cookie).toContain("HttpOnly");
    expect(cookie).toContain("SameSite=Strict");
    expect(cookie).toContain("Path=/");
  });
  it("rejects a bad ?token", async () => {
    const r = await app.request(`/?token=${"0".repeat(token.length)}`);
    expect(r.status).toBe(401);
    expect(r.headers.get("set-cookie")).toBeNull();
  });
  it("401s non-API GETs without a cookie, with a hint", async () => {
    const r = await app.request("/");
    expect(r.status).toBe(401);
    expect(await r.text()).toContain("tokenised URL");
  });
  it("returns 404 JSON when dist is missing and the cookie is valid", async () => {
    const r = await app.request("/", { headers: { Cookie: `dash_token=${token}` } });
    expect(r.status).toBe(404);
    expect(await r.json()).toHaveProperty("error");
  });
});

describe("static serving", () => {
  it("serves files and SPA fallback from webDist", async () => {
    const dist = path.join(stateDir, "dist");
    fs.mkdirSync(path.join(dist, "assets"), { recursive: true });
    fs.writeFileSync(path.join(dist, "index.html"), "<html>spa</html>");
    fs.writeFileSync(path.join(dist, "assets", "a.js"), "console.log(1)");
    const config = { ...loadConfig({}), stateDir, webDist: dist };
    const a = createApp({ config, token, bus: new EventBus() });
    const h = { Cookie: `dash_token=${token}` };
    expect(await (await a.request("/assets/a.js", { headers: h })).text()).toBe("console.log(1)");
    expect(await (await a.request("/some/route", { headers: h })).text()).toBe("<html>spa</html>");
    expect((await a.request("/../../etc/passwd", { headers: h })).status).toBeLessThan(500);
  });
});

describe("EventBus", () => {
  it("delivers events, unsubscribes and reports size", () => {
    const bus = new EventBus();
    const got: unknown[] = [];
    const off = bus.subscribe((e) => got.push(e));
    expect(bus.size).toBe(1);
    bus.emit({ type: "invalidate", resource: "prs" });
    off();
    bus.emit({ type: "invalidate", resource: "ci" });
    expect(got).toEqual([{ type: "invalidate", resource: "prs" }]);
    expect(bus.size).toBe(0);
  });
});
