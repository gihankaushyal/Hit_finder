import type { AgentEventBody } from "../../shared/types";
import type { ApprovalOutcome } from "./approvals";
import type { AgentSessionOptions } from "./policy";
import type { BwrapProfile } from "./bwrap";
import type { ZodType } from "zod";

/** Name of the in-process MCP server that carries the chief of staff's tools. */
export const DASHBOARD_SERVER = "dashboard";
export const mcpToolName = (name: string): string => `mcp__${DASHBOARD_SERVER}__${name}`;

export interface ToolOutput {
  text: string;
  isError?: boolean;
}

/** A tool the dashboard offers to a session; the SDK driver turns it into an MCP tool, the fake driver calls it directly. */
export interface ToolDef {
  name: string;
  description: string;
  schema: Record<string, ZodType>;
  handler(args: Record<string, unknown>): Promise<ToolOutput>;
}

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
  /** Dashboard tools offered to the session (the chief of staff). */
  tools?: ToolDef[];
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
