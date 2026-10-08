import type { Hono } from "hono";
import { streamSSE } from "hono/streaming";
import type { EventBus } from "../events";

/** Idle SSH tunnels and proxies drop quiet connections; a comment line keeps them alive. */
export const HEARTBEAT_MS = 25_000;

/**
 * Hono's stream API has no backpressure signal, so count writes that have not
 * completed. A reader that keeps up has at most one or two in flight; one that
 * stopped reading accumulates them, and is dropped (EventSource reconnects).
 */
export const MAX_PENDING_WRITES = 64;

export function eventRoutes(api: Hono, d: { bus: EventBus; heartbeatMs?: number }): void {
  const heartbeatMs = d.heartbeatMs ?? HEARTBEAT_MS;
  api.get("/events", (c) =>
    streamSSE(c, async (stream) => {
      // Never let a write to a closed stream throw back into bus.emit.
      let pending = 0;
      const unsubscribe = d.bus.subscribe((e) => {
        if (stream.aborted || stream.closed) return;
        if (pending >= MAX_PENDING_WRITES) {
          unsubscribe();
          stream.abort();
          return;
        }
        pending++;
        stream
          .writeSSE({ data: JSON.stringify(e) })
          .catch(() => {})
          .finally(() => {
            pending--;
          });
      });
      stream.onAbort(unsubscribe);
      try {
        await stream.writeSSE({ event: "hello", data: "{}" });
        while (!stream.aborted && !stream.closed) {
          await stream.sleep(heartbeatMs);
          if (stream.aborted || stream.closed) break;
          await stream.write(": ping\n\n");
        }
      } catch {
        // client went away mid-write
      } finally {
        unsubscribe();
      }
    }),
  );
}
