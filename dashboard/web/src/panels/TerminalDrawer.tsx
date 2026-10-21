import { useEffect, useRef, useState } from "react";
import { ArrowsOutLineVertical, X } from "@phosphor-icons/react";
import "@xterm/xterm/css/xterm.css";
import { StatusWord } from "../components/StatusWord";
import { ApiError, apiSend } from "../lib/api";
import { useResource } from "../lib/useResource";
import type { TerminalStatus } from "../../../shared/types";

type Mode = "closed" | "connecting" | "live" | "reconnecting" | "error";

const DEFAULT_BACKOFF_MS = [1000, 2000, 5000, 15000];
const SCROLLBACK_LINES = 2000;
const RESIZE_DEBOUNCE_MS = 100;
const FONT_SIZE_PX = 13;
/** Errors the server gives for a request that will never work as sent; everything else is retried. */
const FATAL_STATUSES = new Set([400, 401, 403, 409, 415]);

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));
const cssVar = (name: string, fallback: string): string => {
  const v = typeof document === "undefined" ? "" : getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
};

async function streamError(res: Response): Promise<ApiError> {
  let message = `The terminal stream failed (HTTP ${res.status}).`;
  try {
    const data = (await res.json()) as { error?: unknown };
    if (typeof data.error === "string" && data.error !== "") message = data.error;
  } catch {
    // not JSON
  }
  return new ApiError(res.status, message);
}

export function TerminalDrawer({ backoffMs = DEFAULT_BACKOFF_MS }: { backoffMs?: number[] }) {
  const status = useResource<TerminalStatus>("terminal", "/api/terminal", () => false);
  const [wanted, setWanted] = useState(false);
  const [mode, setMode] = useState<Mode>("closed");
  const [error, setError] = useState<string | null>(null);
  const [tall, setTall] = useState(false);
  const host = useRef<HTMLDivElement>(null);
  const closeButton = useRef<HTMLButtonElement>(null);
  const openButton = useRef<HTMLButtonElement>(null);
  const off = status.data?.enabled === false;

  useEffect(() => {
    if (!wanted) return;
    let stopped = false;
    let id: string | null = null;
    let live = false;
    let pending = "";
    let sending = false;
    let conn: AbortController | null = null;
    let resizeTimer: ReturnType<typeof setTimeout> | null = null;
    let cleanup: () => void = () => {};
    setMode("connecting");
    setError(null);

    const fail = (message: string) => {
      setError(message);
      setMode("error");
    };

    async function flush(): Promise<void> {
      if (sending || !live || !id) return;
      sending = true;
      try {
        while (pending && live && id) {
          const data = pending;
          pending = "";
          await apiSend("POST", `/api/terminal/${encodeURIComponent(id)}/input`, { data });
        }
      } catch (err) {
        // The server no longer knows this stream: drop it, and the loop below opens a new one.
        if (err instanceof ApiError && (err.status === 404 || err.status === 0)) conn?.abort();
      } finally {
        sending = false;
      }
    }

    void (async () => {
      const [{ Terminal }, { FitAddon }] = await Promise.all([import("@xterm/xterm"), import("@xterm/addon-fit")]);
      if (stopped || !host.current) return;
      const term = new Terminal({
        fontFamily: cssVar("--font-mono", "monospace"),
        fontSize: FONT_SIZE_PX,
        cursorBlink: true,
        scrollback: SCROLLBACK_LINES,
        screenReaderMode: true,
        theme: { background: cssVar("--bg", "#0e1113"), foreground: cssVar("--text", "#e4e7e9"), cursor: cssVar("--accent", "#f0a020") },
      });
      const fit = new FitAddon();
      term.loadAddon(fit);
      term.open(host.current);
      fit.fit();
      term.focus();
      // xterm keeps Tab for the shell, so F6 is the way out of the terminal for keyboard users.
      term.attachCustomKeyEventHandler((e) => {
        if (e.key !== "F6") return true;
        if (e.type === "keydown") closeButton.current?.focus();
        return false;
      });
      term.onData((d) => {
        if (!live) return;
        pending += d;
        void flush();
      });
      term.onResize(({ cols, rows }) => {
        if (resizeTimer) clearTimeout(resizeTimer);
        resizeTimer = setTimeout(() => {
          if (live && id) void apiSend("POST", `/api/terminal/${encodeURIComponent(id)}/resize`, { cols, rows }).catch(() => undefined);
        }, RESIZE_DEBOUNCE_MS);
      });
      const observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(() => fit.fit());
      observer?.observe(host.current);

      cleanup = () => {
        observer?.disconnect();
        if (resizeTimer) clearTimeout(resizeTimer);
        conn?.abort();
        // Ends only this browser's stream; the tmux session, and whatever runs in it, keeps going.
        if (id) void apiSend("DELETE", `/api/terminal/${encodeURIComponent(id)}`).catch(() => undefined);
        term.dispose();
      };

      let attempt = 0;
      while (!stopped) {
        conn = new AbortController();
        try {
          const size = fit.proposeDimensions() ?? { cols: term.cols, rows: term.rows };
          id = (await apiSend<{ id: string }>("POST", "/api/terminal/open", { cols: size.cols, rows: size.rows })).id;
          const res = await fetch(`/api/terminal/${encodeURIComponent(id)}/stream`, { credentials: "same-origin", signal: conn.signal });
          if (!res.ok || !res.body) throw await streamError(res);
          live = true;
          attempt = 0;
          setMode("live");
          const reader = res.body.getReader();
          for (;;) {
            const { done, value } = await reader.read();
            if (done) break;
            term.write(value);
          }
        } catch (err) {
          if (stopped) return;
          if (err instanceof ApiError && FATAL_STATUSES.has(err.status)) {
            live = false;
            return fail(err.message);
          }
        }
        live = false;
        pending = "";
        if (stopped) return;
        setMode("reconnecting");
        await sleep(backoffMs[Math.min(attempt++, backoffMs.length - 1)]);
      }
    })();

    return () => {
      stopped = true;
      live = false;
      cleanup();
    };
  }, [wanted, backoffMs]);

  const close = () => {
    setWanted(false);
    setMode("closed");
    setTimeout(() => openButton.current?.focus(), 0);
  };

  const word =
    mode === "live" ? { kind: "ok" as const, text: "Connected" }
    : mode === "connecting" ? { kind: "running" as const, text: "Connecting" }
    : mode === "reconnecting" ? { kind: "attention" as const, text: "Reconnecting" }
    : mode === "error" ? { kind: "fail" as const, text: "Not connected" }
    : null;

  return (
    <footer className={`drawer${wanted ? " drawer--open" : ""}${tall ? " drawer--tall" : ""}`} aria-labelledby="drawer-title">
      <div className="drawer__bar">
        <h2 id="drawer-title" className="drawer__title">Terminal</h2>
        {word && <StatusWord kind={word.kind}>{word.text}</StatusWord>}
        {off && <span className="dim">The terminal is switched off (DASH_TERMINAL=off).</span>}
        {wanted && <span className="dim drawer__hint">Your shell keeps running in tmux when you close this. F6 leaves the terminal.</span>}
        <span className="topbar__spacer" />
        {wanted && (
          <>
            <button type="button" className="btn btn--quiet" aria-pressed={tall} onClick={() => setTall((v) => !v)}>
              <ArrowsOutLineVertical size={14} aria-hidden="true" /> {tall ? "Smaller" : "Larger"}
            </button>
            <button ref={closeButton} type="button" className="btn" onClick={close}>
              <X size={14} aria-hidden="true" /> Close terminal
            </button>
          </>
        )}
        {!wanted && (
          <button ref={openButton} type="button" className="btn" disabled={off || status.data === null} onClick={() => setWanted(true)}>
            Open terminal
          </button>
        )}
      </div>
      {error && <p className="approval__error" role="alert">{error}</p>}
      {wanted && <div ref={host} className="term" data-testid="terminal" />}
    </footer>
  );
}
