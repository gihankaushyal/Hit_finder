import { describe, it, expect, vi, afterEach } from "vitest";
import { apiGet, apiSend, ApiError, UNAUTHORIZED_MESSAGE } from "../../web/src/lib/api";

const res = (status: number, body: unknown) => ({ ok: status >= 200 && status < 300, status, json: async () => body });
afterEach(() => vi.unstubAllGlobals());

describe("api client", () => {
  it("sends same-origin credentials and parses JSON", async () => {
    const f = vi.fn().mockResolvedValue(res(200, { a: 1 }));
    vi.stubGlobal("fetch", f);
    expect(await apiGet<{ a: number }>("/api/x")).toEqual({ a: 1 });
    expect(f).toHaveBeenCalledWith("/api/x", expect.objectContaining({ credentials: "same-origin", method: "GET" }));
  });

  it("apiSend sends a JSON body", async () => {
    const f = vi.fn().mockResolvedValue(res(200, { ok: true }));
    vi.stubGlobal("fetch", f);
    await apiSend("POST", "/api/y", { t: 1 });
    const init = f.mock.calls[0][1];
    expect(init.method).toBe("POST");
    expect(init.body).toBe('{"t":1}');
    expect(init.credentials).toBe("same-origin");
  });

  it("a non-2xx response throws ApiError carrying the server's error string", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(res(502, { error: "GitHub CLI failed" })));
    await expect(apiGet("/api/x")).rejects.toMatchObject({ name: "ApiError", status: 502, message: "GitHub CLI failed" });
  });

  it("a 401 becomes the clear open-the-tokenised-URL message", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(res(401, { error: "unauthorized" })));
    const err = (await apiGet("/api/x").catch((e: unknown) => e)) as ApiError;
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(401);
    expect(err.message).toBe(UNAUTHORIZED_MESSAGE);
    expect(UNAUTHORIZED_MESSAGE).toMatch(/tokenised URL/);
  });

  it("a network failure becomes a plain message, not a raw TypeError", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    const err = (await apiGet("/api/x").catch((e: unknown) => e)) as ApiError;
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(0);
    expect(err.message).not.toMatch(/TypeError/);
  });

  it("an error body that is not JSON still gives a status message", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 500, json: async () => { throw new Error("x"); } }));
    await expect(apiGet("/api/x")).rejects.toMatchObject({ status: 500, message: "Request failed (HTTP 500)" });
  });
});
