import crypto from "node:crypto";
import fs from "node:fs";
import type { Approval, ApprovalDecision } from "../../shared/types";
import type { EventBus } from "../events";

const SUMMARY_MAX = 200;

export interface ApprovalOutcome {
  decision: ApprovalDecision;
  message?: string;
  /** True when no human decided: the agent stopped or the request was aborted. */
  cancelled?: true;
}

export interface Approvals {
  /** Resolves when a human decides. There is no timeout: nothing is ever decided on its own. */
  request(agent: { id: string; name: string }, tool: string, input: Record<string, unknown>, opts?: { reason?: string; signal?: AbortSignal; onCreated?: (a: Approval) => void }): Promise<ApprovalOutcome>;
  /** Single use: an unknown, decided or cancelled id throws. */
  decide(id: string, decision: ApprovalDecision, message?: string): void;
  cancelAgent(agentId: string): void;
  pending(): Approval[];
  count(): number;
  countFor(agentId: string): number;
}

export function summarise(tool: string, input: Record<string, unknown>): string {
  if (tool === "Bash" && typeof input.command === "string") return input.command.split("\n")[0].slice(0, SUMMARY_MAX);
  if ((tool === "Edit" || tool === "Write" || tool === "NotebookEdit") && typeof input.file_path === "string") return `${tool} ${input.file_path}`.slice(0, SUMMARY_MAX);
  return `${tool} ${JSON.stringify(input)}`.slice(0, SUMMARY_MAX);
}

interface Entry {
  approval: Approval;
  settle: (o: ApprovalOutcome) => void;
  detach: () => void;
}

export function createApprovals(deps: { bus: EventBus; auditPath?: string }): Approvals {
  const entries = new Map<string, Entry>();
  const invalidate = () => deps.bus.emit({ type: "invalidate", resource: "approvals" });

  const audit = (a: Approval, decision: string, message?: string) => {
    if (!deps.auditPath) return;
    const record = { at: new Date().toISOString(), approvalId: a.id, agentId: a.agentId, agentName: a.agentName, tool: a.tool, summary: a.summary, decision, ...(message ? { message } : {}) };
    try {
      fs.appendFileSync(deps.auditPath, JSON.stringify(record) + "\n");
    } catch (err) {
      console.error("approval audit write failed", (err as NodeJS.ErrnoException).code ?? "error");
    }
  };

  const finish = (id: string, outcome: ApprovalOutcome, label: string) => {
    const e = entries.get(id);
    if (!e) return false;
    entries.delete(id);
    e.detach();
    audit(e.approval, label, outcome.message);
    e.settle(outcome);
    invalidate();
    return true;
  };

  const cancelled = (id: string) => finish(id, { decision: "deny", message: "Cancelled", cancelled: true }, "cancelled");

  return {
    request(agent, tool, input, opts = {}) {
      if (opts.signal?.aborted) return Promise.resolve({ decision: "deny", message: "Cancelled", cancelled: true });
      const approval: Approval = {
        id: crypto.randomUUID(),
        agentId: agent.id,
        agentName: agent.name,
        tool,
        input,
        summary: summarise(tool, input),
        reason: opts.reason ?? null,
        requestedAt: new Date().toISOString(),
      };
      return new Promise<ApprovalOutcome>((resolve) => {
        const onAbort = () => void cancelled(approval.id);
        opts.signal?.addEventListener("abort", onAbort, { once: true });
        entries.set(approval.id, { approval, settle: resolve, detach: () => opts.signal?.removeEventListener("abort", onAbort) });
        opts.onCreated?.(approval);
        invalidate();
      });
    },
    decide(id, decision, message) {
      const outcome: ApprovalOutcome = message ? { decision, message } : { decision };
      if (!finish(id, outcome, decision)) throw new Error("Approval not found.");
    },
    cancelAgent(agentId) {
      for (const [id, e] of [...entries]) if (e.approval.agentId === agentId) cancelled(id);
    },
    pending: () => [...entries.values()].map((e) => e.approval).sort((x, y) => x.requestedAt.localeCompare(y.requestedAt)),
    count: () => entries.size,
    countFor: (agentId) => [...entries.values()].filter((e) => e.approval.agentId === agentId).length,
  };
}
