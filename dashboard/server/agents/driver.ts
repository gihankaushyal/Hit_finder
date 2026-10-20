import type { AgentEventBody } from "../../shared/types";
import type { ApprovalOutcome } from "./approvals";
import type { AgentSessionOptions } from "./policy";
import type { BwrapProfile } from "./bwrap";

/** What a session yields: journal-ready events, and the session id once the CLI reports it. */
export type DriverItem = { body: AgentEventBody } | { sessionId: string };

export type PermissionHook = (tool: string, input: Record<string, unknown>, ctx: { signal: AbortSignal; reason?: string }) => Promise<ApprovalOutcome>;

export interface SessionStart {
  prompt: string;
  /** Continue an earlier session with this id. */
  resume?: string;
  options: AgentSessionOptions;
  /** Confinement; absent for the fake driver and for tests. */
  confinement?: Omit<BwrapProfile, "cwd" | "command" | "args">;
  claudeBin: string;
  canUseTool: PermissionHook;
}

export interface Session {
  /** Ends when the session closes. A `result` event ends each turn; the session then waits for `send`. */
  events: AsyncIterable<DriverItem>;
  send(text: string): void;
  /** Stops the current turn; the session stays usable. */
  interrupt(): Promise<void>;
  /** Ends the session and everything it started. */
  close(): void;
}

export interface AgentDriver {
  start(start: SessionStart): Session;
}
