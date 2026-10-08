import { describe, it, expect, beforeEach, vi } from "vitest";
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
  it("tightens loose permissions and keeps a valid token", () => {
    const dir = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "dash-perm-")), "state");
    fs.mkdirSync(dir, { mode: 0o755 });
    fs.chmodSync(dir, 0o755);
    const valid = "ab".repeat(32);
    fs.writeFileSync(path.join(dir, "token"), valid, { mode: 0o644 });
    fs.chmodSync(path.join(dir, "token"), 0o644);
    expect(loadOrCreateToken(dir)).toBe(valid);
    expect(fs.statSync(dir).mode & 0o777).toBe(0o700);
    expect(fs.statSync(path.join(dir, "token")).mode & 0o777).toBe(0o600);
  });
  it("regenerates a malformed stored token", () => {
    const dir = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "dash-bad-")), "state");
    fs.mkdirSync(dir, { mode: 0o700 });
    fs.writeFileSync(path.join(dir, "token"), "short", { mode: 0o600 });
    const t = loadOrCreateToken(dir);
    expect(t).toMatch(/^[0-9a-f]{64}$/);
    expect(fs.readFileSync(path.join(dir, "token"), "utf8").trim()).toBe(t);
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
  });
  it("never serves files outside dist via encoded traversal", async () => {
    const base = fs.mkdtempSync(path.join(os.tmpdir(), "dash-trav-"));
    const dist = path.join(base, "dist");
    fs.mkdirSync(dist);
    fs.writeFileSync(path.join(dist, "index.html"), "<html>spa</html>");
    fs.writeFileSync(path.join(base, "secret.txt"), "SENTINEL-OUTSIDE");
    const config = { ...loadConfig({}), stateDir, webDist: dist };
    const a = createApp({ config, token, bus: new EventBus() });
    const h = { Cookie: `dash_token=${token}` };
    for (const p of ["/%2e%2e/%2e%2e/etc/passwd", "/..%2f..%2fsecret.txt", "/%2e%2e%2fsecret.txt", "/%2e%2e/secret.txt", "/%zz"]) {
      const r = await a.request(p, { headers: h });
      expect(r.status).toBeLessThan(500);
      expect(await r.text()).not.toContain("SENTINEL-OUTSIDE");
    }
  });
  it("does not 500 on a malformed percent-encoded path", async () => {
    const dist = path.join(stateDir, "dist2");
    fs.mkdirSync(dist, { recursive: true });
    fs.writeFileSync(path.join(dist, "index.html"), "<html>spa</html>");
    const config = { ...loadConfig({}), stateDir, webDist: dist };
    const a = createApp({ config, token, bus: new EventBus() });
    const r = await a.request("/%25zz", { headers: { Cookie: `dash_token=${token}` } });
    expect(r.status).toBeLessThan(500);
  });
  it("returns JSON 404 for unknown authenticated /api paths", async () => {
    const dist = path.join(stateDir, "dist3");
    fs.mkdirSync(dist, { recursive: true });
    fs.writeFileSync(path.join(dist, "index.html"), "<html>spa</html>");
    const config = { ...loadConfig({}), stateDir, webDist: dist };
    const a = createApp({ config, token, bus: new EventBus() });
    const r = await a.request("/api/nope", { headers: { Authorization: `Bearer ${token}` } });
    expect(r.status).toBe(404);
    expect(r.headers.get("content-type")).toContain("application/json");
    expect(await r.json()).toEqual({ error: "not found" });
  });
  it("sets nosniff everywhere and no-store on redirect and 401", async () => {
    const ok = await app.request("/api/health", { headers: { Authorization: `Bearer ${token}` } });
    expect(ok.headers.get("x-content-type-options")).toBe("nosniff");
    const redirect = await app.request(`/?token=${token}`);
    expect(redirect.headers.get("cache-control")).toBe("no-store");
    expect(redirect.headers.get("x-content-type-options")).toBe("nosniff");
    const denied = await app.request("/api/health");
    expect(denied.headers.get("cache-control")).toBe("no-store");
    const denied2 = await app.request("/");
    expect(denied2.headers.get("cache-control")).toBe("no-store");
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
  it("isolates throwing listeners", () => {
    const bus = new EventBus();
    const got: unknown[] = [];
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    bus.subscribe(() => {
      throw new Error("bad listener");
    });
    bus.subscribe((e) => got.push(e));
    expect(() => bus.emit({ type: "invalidate", resource: "prs" })).not.toThrow();
    expect(got).toHaveLength(1);
    expect(spy).toHaveBeenCalled();
    spy.mockRestore();
  });
});
