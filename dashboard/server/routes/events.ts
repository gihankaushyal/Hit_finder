import type { Hono } from "hono";
import { streamSSE } from "hono/streaming";
import type { EventBus } from "../events";

/** Idle SSH tunnels and proxies drop quiet connections; a comment line keeps them alive. */
export const HEARTBEAT_MS = 25_000;

/**
 * Each connection owns a queue that one pump drains in order, so a burst
 * emitted in a single tick (pytest output arrives in chunks) is delivered, not
 * dropped. A client is disconnected (EventSource reconnects) only when its
 * queue stays above MAX_QUEUED_EVENTS for QUEUE_STALL_MS: it has stopped reading.
 */
export const MAX_QUEUED_EVENTS = 1000;
export const QUEUE_STALL_MS = 5000;

export function eventRoutes(
  api: Hono,
  d: { bus: EventBus; heartbeatMs?: number; queueLimit?: number; stallMs?: number },
): void {
  const heartbeatMs = d.heartbeatMs ?? HEARTBEAT_MS;
  const queueLimit = d.queueLimit ?? MAX_QUEUED_EVENTS;
  const stallMs = d.stallMs ?? QUEUE_STALL_MS;
  api.get("/events", (c) =>
    streamSSE(c, async (stream) => {
      // Never let a write to a closed stream throw back into bus.emit.
      const queue: string[] = [];
      let pumping = false;
      let overSince: number | null = null;
      const pump = async () => {
        if (pumping) return;
        pumping = true;
        try {
          while (queue.length > 0 && !stream.aborted && !stream.closed) {
            const data = queue.shift() as string;
            await stream.writeSSE({ data }).catch(() => {});
          }
        } finally {
          pumping = false;
        }
      };
      const unsubscribe = d.bus.subscribe((e) => {
        if (stream.aborted || stream.closed) return;
        if (queue.length > queueLimit) {
          const now = Date.now();
          overSince ??= now;
          if (now - overSince >= stallMs) {
            unsubscribe();
            stream.abort();
            return;
          }
        } else {
          overSince = null;
        }
        queue.push(JSON.stringify(e));
        void pump();
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
