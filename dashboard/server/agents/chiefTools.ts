import { z, type ZodType } from "zod";
import { NotImportedError, TaskNotFoundError, type KanbanService } from "../lib/kanbanService";
import type { TaskStatus } from "../../shared/types";
import type { ToolDef, ToolOutput } from "./driver";
import { AgentError, MESSAGE_MAX_CHARS, TASK_MAX_CHARS, type Supervisor } from "./supervisor";

const NAME_PATTERN = /^[a-z0-9][a-z0-9-]{1,30}$/;
const TITLE_MAX_CHARS = 200;
const BODY_MAX_CHARS = 60_000;
const STATUSES = ["todo", "in-progress", "blocked", "done"] as const satisfies readonly TaskStatus[];
const TASK_PREVIEW_CHARS = 80;
const FAILED: ToolOutput = { text: "The request failed. See the dashboard log.", isError: true };

const name = z.string().regex(NAME_PATTERN, "name must be 2 to 31 characters: lowercase letters, digits and hyphens");

/** Wraps a handler so arguments are validated here (the fake driver passes them raw) and errors never leak. */
function def<S extends Record<string, ZodType>>(
  toolName: string,
  description: string,
  schema: S,
  run: (args: z.infer<z.ZodObject<S>>) => Promise<ToolOutput>,
): ToolDef {
  const parser = z.object(schema).strict();
  return {
    name: toolName,
    description,
    schema,
    async handler(raw) {
      const parsed = parser.safeParse(raw);
      if (!parsed.success) return { text: `Invalid arguments: ${parsed.error.issues.map((i) => `${i.path.join(".") || "input"}: ${i.message}`).join("; ")}`, isError: true };
      try {
        return await run(parsed.data);
      } catch (err) {
        if (err instanceof AgentError || err instanceof NotImportedError || err instanceof TaskNotFoundError) return { text: err.message, isError: true };
        console.error(`chief tool ${toolName} failed:`, err instanceof Error ? err.message : err);
        return FAILED;
      }
    },
  };
}

export function createChiefTools(d: { supervisor: Supervisor; kanban: Pick<KanbanService, "addTask" | "setStatus"> }): ToolDef[] {
  const { supervisor, kanban } = d;
  const known = (n: string): ToolOutput | null => (supervisor.get(n) ? null : { text: `Worker "${n}" not found.`, isError: true });
  return [
    def("list_workers", "List the worker agents with their state, branch and pending approvals. Free; use it before acting on a worker.", {}, async () => {
      const all = supervisor.list();
      if (all.length === 0) return { text: "There are no workers." };
      return {
        text: all
          .map((a) => `${a.name} [${a.state}] branch ${a.branch ?? "-"} turns ${a.turns}${a.pendingApprovals ? `, ${a.pendingApprovals} pending approval${a.pendingApprovals === 1 ? "" : "s"}` : ""}: ${a.task.slice(0, TASK_PREVIEW_CHARS)}`)
          .join("\n"),
      };
    }),
    def(
      "spawn_worker",
      "Start a worker agent in its own git worktree on a new branch. The user approves each call. Give it a short lowercase name and a complete, self-contained task.",
      { name, task: z.string().min(1).max(TASK_MAX_CHARS), issue: z.number().int().positive().optional() },
      async (a) => {
        const w = await supervisor.spawn({ name: a.name, task: a.task, issue: a.issue ?? null });
        return { text: `Started worker ${w.name} on branch ${w.branch}.` };
      },
    ),
    def("message_worker", "Send a follow-up message to a running or idle worker. The user approves each call.", { name, text: z.string().min(1).max(MESSAGE_MAX_CHARS) }, async (a) => {
      const missing = known(a.name);
      if (missing) return missing;
      await supervisor.message(a.name, a.text);
      return { text: `Message sent to ${a.name}.` };
    }),
    def("stop_worker", "Stop a worker. Its worktree and branch are kept. The user approves each call.", { name }, async (a) => {
      const missing = known(a.name);
      if (missing) return missing;
      await supervisor.stop(a.name);
      return { text: `Stopped ${a.name}.` };
    }),
    def(
      "create_task",
      "Add an item to the kanban board. Only when the user asked for it. The user approves each call.",
      { title: z.string().min(1).max(TITLE_MAX_CHARS), body: z.string().max(BODY_MAX_CHARS).optional() },
      async (a) => {
        await kanban.addTask(a.title, a.body);
        return { text: `Added task "${a.title}".` };
      },
    ),
    def(
      "update_task",
      "Change a kanban task's status. The user approves each call.",
      { number: z.number().int().positive(), status: z.enum(STATUSES) },
      async (a) => {
        await kanban.setStatus(a.number, a.status);
        return { text: `Task #${a.number} is now ${a.status}.` };
      },
    ),
  ];
}
