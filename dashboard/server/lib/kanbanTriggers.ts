import chokidar from "chokidar";
import type { KanbanService } from "./kanbanService";

/** Background sync cadence; picks up changes made on GitHub. */
export const POLL_INTERVAL_MS = 60_000;

export interface FileWatcher {
  on(event: "add" | "change" | "unlink", fn: () => void): FileWatcher;
  close(): Promise<void> | void;
}
export type WatchFn = (path: string) => FileWatcher;

const defaultWatch: WatchFn = (p) => chokidar.watch(p, { ignoreInitial: true }) as unknown as FileWatcher;

/**
 * Starts the file watcher and the poll timer. Both only call into the service, which ignores
 * them until the first import exists, so this never performs an import by itself.
 */
export function startKanbanTriggers(
  service: KanbanService,
  kanbanPath: string,
  opts: { watch?: WatchFn; pollMs?: number } = {},
): { close(): Promise<void> } {
  const watcher = (opts.watch ?? defaultWatch)(kanbanPath);
  for (const ev of ["add", "change", "unlink"] as const) watcher.on(ev, () => service.fileChanged());
  const timer = setInterval(() => service.poll(), opts.pollMs ?? POLL_INTERVAL_MS);
  timer.unref();
  return {
    async close() {
      clearInterval(timer);
      service.dispose();
      await watcher.close();
    },
  };
}
