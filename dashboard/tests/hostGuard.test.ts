import { describe, it, expect } from "vitest";
import { createApp } from "../server/app";
import { EventBus } from "../server/events";
import type { Config } from "../server/config";

const TOKEN = "h".repeat(64);
const app = createApp({ config: { webDist: "/nonexistent" } as Config, token: TOKEN, bus: new EventBus() });
const auth = { Authorization: `Bearer ${TOKEN}` };

describe("host guard", () => {
  it.each(["http://127.0.0.1:4317", "http://localhost:4317", "http://localhost:8080", "http://[::1]:4317", "http://LOCALHOST:9"])("serves requests addressed to %s (an ssh tunnel may use any local port)", async (origin) => {
    expect((await app.request(`${origin}/api/health`, { headers: auth })).status).toBe(200);
  });

  it.each(["http://evil.example", "http://evil.example:4317", "http://127.0.0.1.evil.example", "http://localhost.evil.example", "http://10.1.2.3:4317", "http://attacker.test:80"])("refuses %s, whatever the token (DNS rebinding)", async (origin) => {
    const api = await app.request(`${origin}/api/health`, { headers: auth });
    expect(api.status).toBe(403);
    expect(await api.text()).not.toContain("ok");
    expect((await app.request(`${origin}/`, { headers: { Cookie: `dash_token=${TOKEN}` } })).status).toBe(403);
    expect((await app.request(`${origin}/?token=${TOKEN}`)).status).toBe(403);
  });

  it("does not reveal whether the token was right", async () => {
    const a = await app.request("http://evil.example/api/health", { headers: auth });
    const b = await app.request("http://evil.example/api/health", { headers: { Authorization: "Bearer wrong" } });
    expect([a.status, await a.text()]).toEqual([b.status, await b.text()]);
  });
});
