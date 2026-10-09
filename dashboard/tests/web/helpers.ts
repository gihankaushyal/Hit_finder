import { vi } from "vitest";

export interface Reply {
  status?: number;
  body?: unknown;
}
export type Handler = Reply | ((body: unknown) => Reply | Promise<Reply>);

export interface Call {
  method: string;
  path: string;
  body: unknown;
}

/**
 * Stubs global fetch at the network boundary. `routes` is keyed "METHOD /path"; a handler may be
 * a fixed reply or a function (so a test can change the answer over time or hold a reply back).
 */
export function mockApi(routes: Record<string, Handler>) {
  const calls: Call[] = [];
  const fetchMock = vi.fn(async (path: string, init?: RequestInit) => {
    const method = init?.method ?? "GET";
    const body = typeof init?.body === "string" ? JSON.parse(init.body) : undefined;
    calls.push({ method, path, body });
    const h = routes[`${method} ${path}`];
    if (!h) return { ok: false, status: 404, json: async () => ({ error: `no route ${method} ${path}` }) };
    const r = typeof h === "function" ? await h(body) : h;
    const status = r.status ?? 200;
    return { ok: status >= 200 && status < 300, status, json: async () => r.body ?? {} };
  });
  vi.stubGlobal("fetch", fetchMock);
  return { calls, fetchMock, count: (method: string, path: string) => calls.filter((c) => c.method === method && c.path === path).length };
}

export class FakeEventSource {
  static instances: FakeEventSource[] = [];
  static get live() {
    return FakeEventSource.instances.filter((i) => !i.closed);
  }
  static emit(data: unknown) {
    for (const i of FakeEventSource.live) i.onmessage?.({ data: JSON.stringify(data) });
  }
  readyState = 1;
  onmessage: ((e: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  onopen: (() => void) | null = null;
  closed = false;
  constructor(public url: string) {
    FakeEventSource.instances.push(this);
  }
  close() {
    this.closed = true;
    this.readyState = 2;
  }
}

export function stubEventSource() {
  FakeEventSource.instances = [];
  vi.stubGlobal("EventSource", FakeEventSource);
}

/** A promise settled from outside, to control the order in which replies arrive. */
export function deferred<T = void>() {
  let resolve!: (v: T) => void;
  const promise = new Promise<T>((r) => (resolve = r));
  return { promise, resolve };
}
