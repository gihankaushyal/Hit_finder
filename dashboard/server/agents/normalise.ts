import type { AgentEventBody } from "../../shared/types";
import type { DriverItem } from "./driver";

export const OUTPUT_MAX = 8000;
const INTERRUPT_NOTICE = "[Request interrupted by user]";

const isObj = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null;

function clip(s: string): string {
  return s.length > OUTPUT_MAX ? `${s.slice(0, OUTPUT_MAX)}\n... (${s.length - OUTPUT_MAX} more characters not shown)` : s;
}

function flatten(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) return content.map((c) => (isObj(c) && typeof c.text === "string" ? c.text : "")).filter(Boolean).join("\n");
  return "";
}

/**
 * Maps one Agent SDK message to zero or more driver items. Thinking blocks, partial stream events,
 * rate-limit and status messages are dropped; anything unrecognised is ignored rather than guessed at.
 */
export function normalise(msg: unknown): DriverItem[] {
  if (!isObj(msg)) return [];
  if (msg.type === "system" && msg.subtype === "init" && typeof msg.session_id === "string") return [{ sessionId: msg.session_id }];

  if (msg.type === "assistant" && isObj(msg.message) && Array.isArray(msg.message.content)) {
    const out: DriverItem[] = [];
    for (const b of msg.message.content) {
      if (!isObj(b)) continue;
      if (b.type === "text" && typeof b.text === "string" && b.text.trim()) out.push({ body: { kind: "text", text: b.text } });
      else if (b.type === "tool_use" && typeof b.id === "string" && typeof b.name === "string") {
        out.push({ body: { kind: "tool-use", tool: b.name, toolUseId: b.id, input: isObj(b.input) ? b.input : {} } });
      }
    }
    return out;
  }

  if (msg.type === "user" && isObj(msg.message) && Array.isArray(msg.message.content)) {
    const out: DriverItem[] = [];
    for (const b of msg.message.content) {
      if (!isObj(b) || b.type !== "tool_result" || typeof b.tool_use_id !== "string") continue;
      const body: AgentEventBody = { kind: "tool-result", toolUseId: b.tool_use_id, output: clip(flatten(b.content)), isError: b.is_error === true };
      out.push({ body });
    }
    return out;
  }

  if (msg.type === "result") {
    const ok = msg.subtype === "success" && msg.is_error !== true;
    const text = typeof msg.result === "string" ? msg.result : ok ? "" : String(msg.subtype ?? "error");
    return [
      {
        body: {
          kind: "result",
          ok,
          text: text === INTERRUPT_NOTICE ? "Interrupted" : text,
          turns: typeof msg.num_turns === "number" ? msg.num_turns : 0,
          costUsd: typeof msg.total_cost_usd === "number" ? msg.total_cost_usd : null,
        },
      },
    ];
  }
  return [];
}
