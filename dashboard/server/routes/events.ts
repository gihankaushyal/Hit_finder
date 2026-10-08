import type { Hono } from "hono";
import { streamSSE } from "hono/streaming";
import type { EventBus } from "../events";

/** Idle SSH tunnels and proxies drop quiet connections; a comment line keeps them alive. */
export const HEARTBEAT_MS = 25_000;

export function eventRoutes(api: Hono, d: { bus: EventBus; heartbeatMs?: number }): void {
  const heartbeatMs = d.heartbeatMs ?? HEARTBEAT_MS;
  api.get("/events", (c) =>
    streamSSE(c, async (stream) => {
      // Never let a write to a closed stream throw back into bus.emit.
      const unsubscribe = d.bus.subscribe((e) => {
        if (stream.aborted || stream.closed) return;
        stream.writeSSE({ data: JSON.stringify(e) }).catch(() => {});
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
