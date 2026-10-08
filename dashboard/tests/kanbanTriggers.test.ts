import { describe, it, expect, vi, afterEach } from "vitest";
import { startKanbanTriggers, type FileWatcher } from "../server/lib/kanbanTriggers";
import type { KanbanService } from "../server/lib/kanbanService";

afterEach(() => vi.useRealTimers());

function fakes() {
  const handlers = new Map<string, () => void>();
  let closed = false;
  const watcher: FileWatcher = {
    on(ev, fn) {
      handlers.set(ev, fn);
      return watcher;
    },
    close() {
      closed = true;
    },
  };
  const calls = { fileChanged: 0, poll: 0, dispose: 0 };
  const service = {
    fileChanged: () => calls.fileChanged++,
    poll: () => calls.poll++,
    dispose: () => calls.dispose++,
  } as unknown as KanbanService;
  return { handlers, watcher, service, calls, isClosed: () => closed };
}

describe("startKanbanTriggers", () => {
  it("routes file events and the poll timer to the service", async () => {
    vi.useFakeTimers();
    const f = fakes();
    const t = startKanbanTriggers(f.service, "/x/kanban.md", { watch: () => f.watcher, pollMs: 1000 });
    for (const ev of ["add", "change", "unlink"]) f.handlers.get(ev)!();
    expect(f.calls.fileChanged).toBe(3);
    await vi.advanceTimersByTimeAsync(3500);
    expect(f.calls.poll).toBe(3);
    await t.close();
  });

  it("close stops the poll, disposes the service and closes the watcher", async () => {
    vi.useFakeTimers();
    const f = fakes();
    const t = startKanbanTriggers(f.service, "/x/kanban.md", { watch: () => f.watcher, pollMs: 1000 });
    await t.close();
    await vi.advanceTimersByTimeAsync(5000);
    expect(f.calls.poll).toBe(0);
    expect(f.calls.dispose).toBe(1);
    expect(f.isClosed()).toBe(true);
  });
});
