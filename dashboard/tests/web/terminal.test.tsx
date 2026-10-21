import { describe, it, expect, afterEach, beforeEach, vi } from "vitest";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

// xterm needs a real canvas and layout; the component's own logic is what is tested here.
const terms: FakeTerm[] = [];
class FakeTerm {
  cols = 100;
  rows = 14;
  written: Uint8Array[] = [];
  disposed = false;
  dataCb: (d: string) => void = () => {};
  resizeCb: (s: { cols: number; rows: number }) => void = () => {};
  keyHandler: (e: KeyboardEvent) => boolean = () => true;
  options: Record<string, unknown> = {};
  constructor(opts: Record<string, unknown>) {
    this.options = opts;
    terms.push(this);
  }
  loadAddon() {}
  open() {}
  write(d: Uint8Array) { this.written.push(d); }
  focus() {}
  dispose() { this.disposed = true; }
  onData(cb: (d: string) => void) { this.dataCb = cb; return { dispose() {} }; }
  onResize(cb: (s: { cols: number; rows: number }) => void) { this.resizeCb = cb; return { dispose() {} }; }
  attachCustomKeyEventHandler(fn: (e: KeyboardEvent) => boolean) { this.keyHandler = fn; }
}
vi.mock("@xterm/xterm", () => ({ Terminal: FakeTerm }));
vi.mock("@xterm/xterm/css/xterm.css", () => ({}));
vi.mock("@xterm/addon-fit", () => ({ FitAddon: class { fit() {} proposeDimensions() { return { cols: 100, rows: 14 }; } } }));

import { TerminalDrawer } from "../../web/src/panels/TerminalDrawer";
import { stubEventSource } from "./helpers";

interface Stream { push(b: Uint8Array): void; end(): void }
let streams: Stream[];
let calls: { method: string; path: string; body?: unknown }[];
let opens: number;
let status: { enabled: boolean; max: number; open: number };
let openReply: { status: number; body: unknown };
let inputStatus: number;

function installFetch() {
  vi.stubGlobal("fetch", vi.fn(async (path: string, init?: RequestInit) => {
    const method = init?.method ?? "GET";
    const body = typeof init?.body === "string" ? JSON.parse(init.body) : undefined;
    calls.push({ method, path, body });
    const json = (s: number, b: unknown) => ({ ok: s < 300, status: s, json: async () => b });
    if (method === "GET" && path === "/api/terminal") return json(200, status);
    if (method === "POST" && path === "/api/terminal/open") {
      opens++;
      return json(openReply.status, openReply.status < 300 ? { id: `id-${opens}` } : openReply.body);
    }
    if (method === "GET" && path.endsWith("/stream")) {
      let ctl!: ReadableStreamDefaultController<Uint8Array>;
      const rs = new ReadableStream<Uint8Array>({ start: (c) => (ctl = c) });
      streams.push({ push: (b) => ctl.enqueue(b), end: () => ctl.close() });
      init?.signal?.addEventListener("abort", () => { try { ctl.error(new DOMException("aborted", "AbortError")); } catch { /* closed */ } });
      return { ok: true, status: 200, body: rs, json: async () => ({}) };
    }
    if (method === "POST" && path.endsWith("/input")) return json(inputStatus, inputStatus < 300 ? { ok: true } : { error: "Terminal not found. Open it again." });
    return json(200, { ok: true });
  }));
}
const posts = (suffix: string) => calls.filter((c) => c.method === "POST" && c.path.endsWith(suffix));

beforeEach(() => {
  stubEventSource();
  terms.length = 0;
  streams = [];
  calls = [];
  opens = 0;
  status = { enabled: true, max: 2, open: 0 };
  openReply = { status: 201, body: {} };
  inputStatus = 200;
  installFetch();
});
afterEach(() => vi.unstubAllGlobals());

const openIt = async () => userEvent.click(await screen.findByRole("button", { name: "Open terminal" }));

describe("TerminalDrawer", () => {
  it("is a collapsed bar until opened, and makes no stream before that", async () => {
    render(<TerminalDrawer backoffMs={[5]} />);
    expect(await screen.findByRole("button", { name: "Open terminal" })).toBeEnabled();
    expect(screen.getByRole("heading", { name: "Terminal" })).toBeInTheDocument();
    expect(posts("/open")).toHaveLength(0);
    expect(terms).toHaveLength(0);
  });

  it("says so, and cannot be opened, when the terminal is switched off", async () => {
    status = { enabled: false, max: 2, open: 0 };
    render(<TerminalDrawer backoffMs={[5]} />);
    expect(await screen.findByText(/switched off/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Open terminal" })).toBeDisabled();
  });

  it("opens a stream at the fitted size and writes what the shell sends, bytes untouched", async () => {
    render(<TerminalDrawer backoffMs={[5]} />);
    await openIt();
    await waitFor(() => expect(streams).toHaveLength(1));
    expect(posts("/open")[0].body).toEqual({ cols: 100, rows: 14 });
    expect(calls.find((c) => c.path === "/api/terminal/id-1/stream")).toBeTruthy();
    act(() => streams[0].push(new Uint8Array([0x68, 0xc3])));
    act(() => streams[0].push(new Uint8Array([0xa9])));
    await waitFor(() => expect(terms[0].written).toHaveLength(2));
    expect(Buffer.concat(terms[0].written).toString("utf8")).toBe("hé");
    expect(await screen.findByText("Connected")).toBeInTheDocument();
  });

  it("sends keystrokes in order, joining ones typed while a request is in flight", async () => {
    render(<TerminalDrawer backoffMs={[5]} />);
    await openIt();
    await waitFor(() => expect(streams).toHaveLength(1));
    act(() => { terms[0].dataCb("l"); terms[0].dataCb("s"); terms[0].dataCb("\r"); });
    await waitFor(() => expect(posts("/input").map((c) => (c.body as { data: string }).data).join("")).toBe("ls\r"));
    act(() => terms[0].dataCb("x"));
    await waitFor(() => expect(posts("/input").map((c) => (c.body as { data: string }).data).join("")).toBe("ls\rx"));
  });

  it("tells the server when the terminal is resized", async () => {
    render(<TerminalDrawer backoffMs={[5]} />);
    await openIt();
    await waitFor(() => expect(streams).toHaveLength(1));
    act(() => terms[0].resizeCb({ cols: 120, rows: 40 }));
    await waitFor(() => expect(posts("/resize")[0]?.body).toEqual({ cols: 120, rows: 40 }));
  });

  it("reconnects when the stream drops, and shows that it is doing so", async () => {
    render(<TerminalDrawer backoffMs={[5]} />);
    await openIt();
    await waitFor(() => expect(streams).toHaveLength(1));
    act(() => streams[0].end());
    await waitFor(() => expect(streams).toHaveLength(2));
    expect(posts("/open")).toHaveLength(2);
    expect(terms).toHaveLength(1); // the same terminal window keeps its screen
  });

  it("reconnects when the server no longer knows the stream", async () => {
    render(<TerminalDrawer backoffMs={[5]} />);
    await openIt();
    await waitFor(() => expect(streams).toHaveLength(1));
    inputStatus = 404;
    act(() => terms[0].dataCb("a"));
    await waitFor(() => expect(streams).toHaveLength(2));
  });

  it("explains a refusal such as the stream limit and does not retry in a loop", async () => {
    openReply = { status: 409, body: { error: "2 terminal streams are already open. Close one first." } };
    render(<TerminalDrawer backoffMs={[5]} />);
    await openIt();
    expect(await screen.findByRole("alert")).toHaveTextContent("2 terminal streams are already open");
    await new Promise((r) => setTimeout(r, 60));
    expect(posts("/open")).toHaveLength(1);
  });

  it("Close terminal ends the stream on the server, disposes the screen and collapses", async () => {
    render(<TerminalDrawer backoffMs={[5]} />);
    await openIt();
    await waitFor(() => expect(streams).toHaveLength(1));
    await userEvent.click(screen.getByRole("button", { name: "Close terminal" }));
    await waitFor(() => expect(calls.some((c) => c.method === "DELETE" && c.path === "/api/terminal/id-1")).toBe(true));
    expect(terms[0].disposed).toBe(true);
    expect(await screen.findByRole("button", { name: "Open terminal" })).toBeInTheDocument();
    await new Promise((r) => setTimeout(r, 60));
    expect(posts("/open")).toHaveLength(1); // closing is not a drop
  });

  it("F6 moves focus out of the terminal to the Close button", async () => {
    render(<TerminalDrawer backoffMs={[5]} />);
    await openIt();
    await waitFor(() => expect(streams).toHaveLength(1));
    const handled = terms[0].keyHandler(new KeyboardEvent("keydown", { key: "F6" }));
    expect(handled).toBe(false); // not passed on to the shell
    expect(screen.getByRole("button", { name: "Close terminal" })).toHaveFocus();
    expect(terms[0].keyHandler(new KeyboardEvent("keydown", { key: "a" }))).toBe(true);
    expect(screen.getByText(/F6/)).toBeInTheDocument();
  });

  it("closes its stream when the page component goes away", async () => {
    const { unmount } = render(<TerminalDrawer backoffMs={[5]} />);
    await openIt();
    await waitFor(() => expect(streams).toHaveLength(1));
    unmount();
    await waitFor(() => expect(calls.some((c) => c.method === "DELETE")).toBe(true));
    expect(terms[0].disposed).toBe(true);
  });

  it("turns on screen-reader support and a scrollback limit", async () => {
    render(<TerminalDrawer backoffMs={[5]} />);
    await openIt();
    await waitFor(() => expect(terms).toHaveLength(1));
    expect(terms[0].options).toMatchObject({ screenReaderMode: true, scrollback: 2000 });
  });
});
