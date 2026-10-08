import { describe, it, expect, afterEach, beforeEach, vi } from "vitest";
import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Kanban, DONE_VISIBLE_COUNT } from "../../web/src/panels/Kanban";
import { FakeEventSource, deferred, mockApi, stubEventSource, type Handler } from "./helpers";
import type { Board, Task, TaskStatus } from "../../shared/types";

const task = (n: number, title: string, status: TaskStatus, over: Partial<Task> = {}): Task => ({
  number: n, title, status, kind: null, url: `https://github.com/o/r/issues/${n}`,
  updatedAt: `2026-10-0${Math.min(n, 9)}T00:00:00Z`, inFile: true, ...over,
});
const emptyBoard = (): Board => ({
  columns: { todo: [], "in-progress": [], blocked: [], done: [] },
  conflicts: [], lastSyncAt: "2026-10-07T10:00:00Z", syncError: null, imported: true,
});
const boardOf = (tasks: Task[], over: Partial<Board> = {}): Board => {
  const b = emptyBoard();
  for (const t of tasks) b.columns[t.status].push(t);
  return { ...b, ...over };
};
const standard = () => [task(1, "Write docs", "todo", { kind: "feature" }), task(2, "Fix loader", "in-progress"), task(3, "Wait for GPU", "blocked"), task(4, "Ship cache", "done")];

/** A tiny stand-in server: PATCH changes the stored status, GET returns what is stored. */
function server(initial: Board, extra: Record<string, Handler> = {}) {
  const state = { board: initial };
  const api = mockApi({
    "GET /api/kanban": () => ({ body: state.board }),
    "PATCH /api/kanban/tasks/1": (b) => apply(1, b),
    "PATCH /api/kanban/tasks/2": (b) => apply(2, b),
    "POST /api/kanban/tasks": { status: 201, body: { ok: true } },
    ...extra,
  });
  function apply(n: number, b: unknown) {
    const status = (b as { status: TaskStatus }).status;
    const all = Object.values(state.board.columns).flat();
    state.board = boardOf(all.map((t) => (t.number === n ? { ...t, status } : t)), { imported: true });
    return { body: { ok: true } };
  }
  return { state, api };
}
const column = (name: string) => screen.getByRole("region", { name: new RegExp(`^${name}`) });

beforeEach(() => stubEventSource());
afterEach(() => vi.unstubAllGlobals());

describe("Kanban states", () => {
  it("loading shows a skeleton", () => {
    mockApi({ "GET /api/kanban": () => new Promise(() => {}) as never });
    render(<Kanban />);
    expect(screen.getByText("Loading Kanban")).toBeInTheDocument();
  });

  it("error shows the server's message and Retry reloads", async () => {
    let n = 0;
    const api = mockApi({ "GET /api/kanban": () => (n++ === 0 ? { status: 502, body: { error: "gh exited with code 1" } } : { body: boardOf(standard()) }) });
    render(<Kanban />);
    expect(await screen.findByRole("alert")).toHaveTextContent("gh exited with code 1");
    await userEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByText("Write docs")).toBeInTheDocument();
    expect(api.count("GET", "/api/kanban")).toBe(2);
  });

  it("empty imported board says what to do and still offers the add form", async () => {
    server(emptyBoard());
    render(<Kanban />);
    expect(await screen.findByText(/No tasks yet\. Add one in the Todo column\./)).toBeInTheDocument();
    expect(screen.getByLabelText("Task")).toBeInTheDocument();
  });
});

describe("Kanban columns", () => {
  it("puts tasks in the right columns with counts, number links, kind and not-in-file chips", async () => {
    server(boardOf([task(1, "Write docs", "todo", { kind: "feature", inFile: false }), ...standard().slice(1)]));
    render(<Kanban />);
    await screen.findByText("Write docs");
    for (const [name, title, count] of [["Todo", "Write docs", "1"], ["In progress", "Fix loader", "1"], ["Blocked", "Wait for GPU", "1"], ["Done", "Ship cache", "1"]]) {
      const col = column(name);
      expect(within(col).getByText(title)).toBeInTheDocument();
      expect(within(col).getByRole("heading", { level: 3 })).toHaveTextContent(`${name} ${count}`);
    }
    const todo = column("Todo");
    expect(within(todo).getByText("feature")).toHaveClass("chip");
    expect(within(todo).getByText("not in file")).toBeInTheDocument();
    const link = within(todo).getByRole("link", { name: /Issue #1/ });
    expect(link).toHaveTextContent("#1");
    expect(link).toHaveClass("mono");
    expect(link).toHaveAttribute("href", "https://github.com/o/r/issues/1");
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", "noopener noreferrer");
  });

  it("a Done task title is struck through in an s element and its box is ticked", async () => {
    server(boardOf(standard()));
    render(<Kanban />);
    const title = await screen.findByText("Ship cache");
    expect(title.closest("s")).not.toBeNull();
    expect(screen.getByRole("checkbox", { name: "Mark Ship cache done" })).toBeChecked();
    expect(screen.getByText("Write docs").closest("s")).toBeNull();
  });

  it("caps the Done column and reveals the rest on request", async () => {
    const many = Array.from({ length: DONE_VISIBLE_COUNT + 3 }, (_, i) => task(100 + i, `Done ${i}`, "done", { updatedAt: new Date(2026, 9, 1, 0, 0, 100 - i).toISOString() }));
    server(boardOf(many));
    render(<Kanban />);
    await screen.findByText("Done 0");
    expect(within(column("Done")).getAllByRole("listitem")).toHaveLength(DONE_VISIBLE_COUNT);
    await userEvent.click(screen.getByRole("button", { name: `Show all (${DONE_VISIBLE_COUNT + 3})` }));
    expect(within(column("Done")).getAllByRole("listitem")).toHaveLength(DONE_VISIBLE_COUNT + 3);
  });
});

describe("Kanban before the first import", () => {
  it("is read-only with one notice and the exact command, and no controls that would return 409", async () => {
    const api = server(boardOf([task(-1, "From the file", "todo", { url: "" }), task(-2, "Checked in file", "done", { url: "" })], { imported: false, lastSyncAt: null })).api;
    render(<Kanban />);
    await screen.findByText("From the file");
    const notice = screen.getByRole("note");
    expect(notice).toHaveTextContent(/read from the kanban file/i);
    expect(within(notice).getByText("npm run kanban:sync -- --dry-run")).toHaveClass("mono");
    expect(screen.queryByRole("checkbox")).not.toBeInTheDocument();
    expect(screen.queryByRole("combobox")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Task")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Add task" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Sync now/ })).not.toBeInTheDocument();
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
    expect(screen.getByText("Checked in file").closest("s")).not.toBeNull();
    expect(api.calls.filter((c) => c.method !== "GET")).toHaveLength(0);
  });
});

describe("Kanban changes", () => {
  it("ticking a Todo task PATCHes done and moves it to Done at once, before the reply", async () => {
    const gate = deferred();
    const { api, state } = server(boardOf(standard()), {
      "PATCH /api/kanban/tasks/1": async (b) => { await gate.promise; state.board = boardOf(standard().map((t) => (t.number === 1 ? { ...t, status: (b as { status: TaskStatus }).status } : t))); return { body: { ok: true } }; },
    });
    render(<Kanban />);
    await userEvent.click(await screen.findByRole("checkbox", { name: "Mark Write docs done" }));
    expect(within(column("Done")).getByText("Write docs").closest("s")).not.toBeNull();
    expect(within(column("Todo")).queryByText("Write docs")).not.toBeInTheDocument();
    expect(api.calls.find((c) => c.method === "PATCH")).toMatchObject({ path: "/api/kanban/tasks/1", body: { status: "done" } });
    gate.resolve();
    await waitFor(() => expect(api.count("GET", "/api/kanban")).toBeGreaterThan(1));
    await waitFor(() => expect(within(column("Done")).getByText("Write docs")).toBeInTheDocument());
    // keyboard focus follows the control to its new column
    expect(screen.getByRole("checkbox", { name: "Mark Write docs done" })).toHaveFocus();
  });

  it("unticking a Done task reopens it to todo", async () => {
    const { api } = server(boardOf(standard()), { "PATCH /api/kanban/tasks/4": { body: { ok: true } } });
    render(<Kanban />);
    await userEvent.click(await screen.findByRole("checkbox", { name: "Mark Ship cache done" }));
    expect(within(column("Todo")).getByText("Ship cache")).toBeInTheDocument();
    expect(api.calls.find((c) => c.method === "PATCH")?.body).toEqual({ status: "todo" });
  });

  it("a failed PATCH moves the task back and an alert names it", async () => {
    server(boardOf(standard()), { "PATCH /api/kanban/tasks/1": { status: 502, body: { error: "gh failed" } } });
    render(<Kanban />);
    await userEvent.click(await screen.findByRole("checkbox", { name: "Mark Write docs done" }));
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Write docs");
    expect(alert).toHaveTextContent("gh failed");
    expect(within(column("Todo")).getByText("Write docs")).toBeInTheDocument();
    expect(within(column("Done")).queryByText("Write docs")).not.toBeInTheDocument();
  });

  it("the Move select is the keyboard path and PATCHes the chosen status", async () => {
    const { api } = server(boardOf(standard()));
    render(<Kanban />);
    const sel = await screen.findByRole("combobox", { name: "Move Write docs" });
    expect(sel).toHaveValue("todo");
    await userEvent.selectOptions(sel, "blocked");
    expect(api.calls.find((c) => c.method === "PATCH")).toMatchObject({ path: "/api/kanban/tasks/1", body: { status: "blocked" } });
    expect(within(column("Blocked")).getByText("Write docs")).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "Move Write docs" })).toHaveFocus();
  });

  it("two quick changes to one task, with replies out of order, end in the last chosen column", async () => {
    const first = deferred();
    const second = deferred();
    const gates = [first, second];
    const sent: string[] = [];
    const { state } = server(boardOf(standard()), {
      "PATCH /api/kanban/tasks/1": async (b) => {
        const status = (b as { status: TaskStatus }).status;
        sent.push(status);
        await gates[sent.length - 1].promise;
        state.board = boardOf(standard().map((t) => (t.number === 1 ? { ...t, status } : t)));
        return { body: { ok: true } };
      },
    });
    render(<Kanban />);
    await userEvent.click(await screen.findByRole("checkbox", { name: "Mark Write docs done" }));
    await userEvent.click(screen.getByRole("checkbox", { name: "Mark Write docs done" })); // untick at once
    expect(within(column("Todo")).getByText("Write docs")).toBeInTheDocument();
    // the later reply cannot arrive first: the second request waits for the first
    second.resolve();
    first.resolve();
    await waitFor(() => expect(sent).toEqual(["done", "todo"]));
    await waitFor(() => expect(state.board.columns.todo.map((t) => t.title)).toContain("Write docs"));
    await act(async () => { FakeEventSource.emit({ type: "invalidate", resource: "kanban" }); });
    await waitFor(() => expect(within(column("Todo")).getByText("Write docs")).toBeInTheDocument());
    expect(within(column("Done")).queryByText("Write docs")).not.toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("a stale failure of an earlier change does not roll back a later one", async () => {
    let call = 0;
    const { state } = server(boardOf(standard()), {
      "PATCH /api/kanban/tasks/1": async (b) => {
        const status = (b as { status: TaskStatus }).status;
        if (call++ === 0) return { status: 502, body: { error: "first failed" } };
        state.board = boardOf(standard().map((t) => (t.number === 1 ? { ...t, status } : t)));
        return { body: { ok: true } };
      },
    });
    render(<Kanban />);
    await userEvent.click(await screen.findByRole("checkbox", { name: "Mark Write docs done" }));
    await userEvent.selectOptions(screen.getByRole("combobox", { name: "Move Write docs" }), "blocked");
    await waitFor(() => expect(state.board.columns.blocked.map((t) => t.title)).toContain("Write docs"));
    expect(within(column("Blocked")).getByText("Write docs")).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});

describe("Kanban add task", () => {
  it("POSTs the trimmed title, clears the field and keeps focus there", async () => {
    const { api } = server(boardOf(standard()));
    render(<Kanban />);
    const input = await screen.findByLabelText("Task");
    await userEvent.type(input, "  New idea  {Enter}");
    await waitFor(() => expect(api.calls.find((c) => c.method === "POST")?.body).toEqual({ title: "New idea" }));
    await waitFor(() => expect(input).toHaveValue(""));
    expect(input).toHaveFocus();
    await waitFor(() => expect(api.count("GET", "/api/kanban")).toBeGreaterThan(1));
  });

  it("ignores whitespace-only input without calling the server", async () => {
    const { api } = server(boardOf(standard()));
    render(<Kanban />);
    const input = await screen.findByLabelText("Task");
    await userEvent.type(input, "   {Enter}");
    expect(api.calls.filter((c) => c.method === "POST")).toHaveLength(0);
    expect(screen.getByText("Enter a task title.")).toBeInTheDocument();
  });

  it("shows the server's 400 next to the field and keeps the text", async () => {
    server(boardOf(standard()), { "POST /api/kanban/tasks": { status: 400, body: { error: "title must be 1 to 200 characters" } } });
    render(<Kanban />);
    const input = await screen.findByLabelText("Task");
    await userEvent.type(input, "x{Enter}");
    expect(await screen.findByText("title must be 1 to 200 characters")).toBeInTheDocument();
    expect(input).toHaveValue("x");
    expect(input).toHaveAccessibleDescription(/title must be 1 to 200/);
  });
});

describe("Kanban sync and notices", () => {
  it("renders conflicts inside a details element and sync errors as a failure", async () => {
    server(boardOf(standard(), { conflicts: ["#1 edited in both places"], syncError: "sync failed; see the server log" }));
    render(<Kanban />);
    const conflict = await screen.findByText("#1 edited in both places");
    expect(conflict.closest("details")).not.toBeNull();
    expect(screen.getByText("sync failed; see the server log")).toBeInTheDocument();
  });

  it("shows a conflict next to the task it concerns as well as in the notice", async () => {
    const msgs = ["#1 changed in both places; kept GitHub state", "creating \"Other\" failed; the next sync will retry"];
    server(boardOf(standard(), { conflicts: msgs, conflictItems: [{ issue: 1, message: msgs[0] }, { issue: null, message: msgs[1] }] }));
    render(<Kanban />);
    const row = (await screen.findByText("Write docs")).closest("li")!;
    expect(within(row).getByText(msgs[0])).toBeInTheDocument();
    // the other tasks carry no conflict text
    expect(within(screen.getByText("Fix loader").closest("li")!).queryByText(/changed in both/)).toBeNull();
    // the notice still lists every conflict, including the one with no task
    const notice = screen.getByText("2 conflicts need a look").closest("details")!;
    expect(within(notice).getByText(msgs[0])).toBeInTheDocument();
    expect(within(notice).getByText(msgs[1])).toBeInTheDocument();
    expect(screen.queryAllByText(msgs[1])).toHaveLength(1);
  });

  it("a board without the structured list still shows its conflicts in the notice", async () => {
    server(boardOf(standard(), { conflicts: ["#1 edited in both places"] }));
    render(<Kanban />);
    expect(await screen.findByText("#1 edited in both places")).toBeInTheDocument();
    expect(within(screen.getByText("Write docs").closest("li")!).queryByText(/edited in both/)).toBeNull();
  });

  it("Sync now posts to the sync route and shows that it is running", async () => {
    const gate = deferred();
    const { api } = server(boardOf(standard()), {
      "POST /api/kanban/sync": async () => { await gate.promise; return { body: boardOf(standard()) }; },
    });
    render(<Kanban />);
    await userEvent.click(await screen.findByRole("button", { name: "Sync now" }));
    expect(api.count("POST", "/api/kanban/sync")).toBe(1);
    expect(screen.getByRole("button", { name: "Syncing" })).toBeDisabled();
    gate.resolve();
    expect(await screen.findByRole("button", { name: "Sync now" })).toBeEnabled();
  });
});
