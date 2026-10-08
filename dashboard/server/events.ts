import type { DashEvent } from "../shared/types";

type Listener = (e: DashEvent) => void;

export class EventBus {
  private listeners = new Set<Listener>();

  get size(): number {
    return this.listeners.size;
  }

  emit(e: DashEvent): void {
    for (const fn of [...this.listeners]) {
      try {
        fn(e);
      } catch (err) {
        console.error("event listener failed", err);
      }
    }
  }

  subscribe(fn: Listener): () => void {
    this.listeners.add(fn);
    return () => {
      this.listeners.delete(fn);
    };
  }
}
