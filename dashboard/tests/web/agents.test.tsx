import { describe, it, expect, afterEach, beforeEach, vi } from "vitest";
import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Agents } from "../../web/src/panels/Agents";
import { FakeEventSource, mockApi, stubEventSource } from "./helpers";
import type { AgentEvent, AgentEventBody, AgentEventsResponse, AgentState, AgentSummary, AgentsResponse } from "../../shared/types";

beforeEach(() => stubEventSource());
afterEach(() => vi.unstubAllGlobals());

const agent = (name: string, state: AgentState = "idle", over: Partial<AgentSummary> = {}): AgentSummary => ({
  id: name, name, role: "worker", task: `task for ${name}`, issue: null, state, model: "sonnet", branch: `phase-05-agent-${name}`,
  worktree: `/w/${name}`, createdAt: "2026-10-20T08:00:00.000Z", updatedAt: "2026-10-20T08:05:00.000Z", lastActivity: null, turns: 1, costUsd: null,
  sessionId: "s-1", error: null, pendingApprovals: 0, lastSeq: 3, ...over,
});
const list = (agents: AgentSummary[]): { body: AgentsResponse } => ({ body: { agents, max: 3, driver: "sdk" } });
const ev = (seq: number, e: AgentEventBody): AgentEvent => ({ ...e, seq, at: "2026-10-20T08:01:00.000Z" });
const log = (events: AgentEvent[], lastSeq = events.length): { body: AgentEventsResponse } => ({ body: { events, lastSeq } });

describe("Agents list", () => {
  it("shows an empty state that says how to start one", async () => {
    mockApi({ "GET /api/agents": list([]) });
    render(<Agents />);
    expect(await screen.findByText(/No agents yet/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "New agent" })).toBeInTheDocument();
  });

  it("shows each agent with its name, task, an icon-plus-word state and turns", async () => {
    mockApi({ "GET /api/agents": list([agent("fix-lcn", "running"), agent("write-docs", "failed", { error: "The agent session ended with an error." })]) });
    render(<Agents />);
    const rows = await screen.findAllByRole("listitem");
    expect(rows).toHaveLength(2);
    expect(within(rows[0]).getByText("fix-lcn")).toHaveClass("mono");
    expect(within(rows[0]).getByText("task for fix-lcn")).toBeInTheDocument();
    expect(within(rows[0]).getByText("Running")).toBeInTheDocument();
    expect(within(rows[1]).getByText("Failed")).toBeInTheDocument();
    expect(within(rows[1]).getByText("The agent session ended with an error.")).toBeInTheDocument();
  });

  it("flags an agent that is waiting for approval with a link to the cards", async () => {
    mockApi({ "GET /api/agents": list([agent("pusher", "waiting-approval", { pendingApprovals: 2 })]) });
    render(<Agents />);
    const row = await screen.findByRole("listitem");
    expect(within(row).getByText("Waiting for approval")).toBeInTheDocument();
    expect(within(row).getByRole("link", { name: /2 pending/ })).toHaveAttribute("href", "#approvals");
  });

  it("shows how many of the allowed agents are in use", async () => {
    mockApi({ "GET /api/agents": list([agent("a-one", "running"), agent("b-two", "stopped")]) });
    render(<Agents />);
    expect(await screen.findByText("1 / 3 running")).toBeInTheDocument();
  });

  it("refetches when the server announces a change", async () => {
    let agents = [agent("a-one")];
    const api = mockApi({ "GET /api/agents": () => list(agents) });
    render(<Agents />);
    await screen.findByText("a-one");
    agents = [agent("a-one"), agent("b-two")];
    act(() => FakeEventSource.emit({ type: "invalidate", resource: "agents" }));
    expect(await screen.findByText("b-two")).toBeInTheDocument();
    expect(api.count("GET", "/api/agents")).toBeGreaterThan(1);
  });

  it("shows the error state with a retry", async () => {
    mockApi({ "GET /api/agents": { status: 500, body: { error: "internal error" } } });
    render(<Agents />);
    expect(await screen.findByRole("alert")).toHaveTextContent("internal error");
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });
});

describe("New agent", () => {
  async function open() {
    await userEvent.click(await screen.findByRole("button", { name: "New agent" }));
  }

  it("rejects a bad name or an empty task without calling the server", async () => {
    const api = mockApi({ "GET /api/agents": list([]) });
    render(<Agents />);
    await open();
    await userEvent.type(screen.getByLabelText("Name"), "Bad Name");
    await userEvent.type(screen.getByLabelText("Task"), "do it");
    await userEvent.click(screen.getByRole("button", { name: "Start agent" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(/lowercase letters/);
    await userEvent.clear(screen.getByLabelText("Name"));
    await userEvent.type(screen.getByLabelText("Name"), "good-name");
    await userEvent.clear(screen.getByLabelText("Task"));
    await userEvent.click(screen.getByRole("button", { name: "Start agent" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(/task/i);
    expect(api.count("POST", "/api/agents")).toBe(0);
  });

  it("starts an agent with the name, task and issue, then closes the form", async () => {
    const api = mockApi({ "GET /api/agents": list([]), "POST /api/agents": { status: 201, body: agent("fix-lcn", "starting") } });
    render(<Agents />);
    await open();
    await userEvent.type(screen.getByLabelText("Name"), "fix-lcn");
    await userEvent.type(screen.getByLabelText("Task"), "tidy the notes");
    await userEvent.type(screen.getByLabelText(/Issue/), "12");
    await userEvent.click(screen.getByRole("button", { name: "Start agent" }));
    await waitFor(() => expect(api.calls.find((c) => c.method === "POST")?.body).toEqual({ name: "fix-lcn", task: "tidy the notes", issue: 12 }));
    await waitFor(() => expect(screen.queryByLabelText("Name")).toBeNull());
  });

  it("omits the issue when it is blank", async () => {
    const api = mockApi({ "GET /api/agents": list([]), "POST /api/agents": { status: 201, body: agent("fix-lcn", "starting") } });
    render(<Agents />);
    await open();
    await userEvent.type(screen.getByLabelText("Name"), "fix-lcn");
    await userEvent.type(screen.getByLabelText("Task"), "tidy");
    await userEvent.click(screen.getByRole("button", { name: "Start agent" }));
    await waitFor(() => expect(api.calls.find((c) => c.method === "POST")?.body).toEqual({ name: "fix-lcn", task: "tidy" }));
  });

  it("shows the server's refusal, such as the cap, and keeps what was typed", async () => {
    mockApi({ "GET /api/agents": list([]), "POST /api/agents": { status: 409, body: { error: "3 agents are already running. Stop one first." } } });
    render(<Agents />);
    await open();
    await userEvent.type(screen.getByLabelText("Name"), "fix-lcn");
    await userEvent.type(screen.getByLabelText("Task"), "tidy");
    await userEvent.click(screen.getByRole("button", { name: "Start agent" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("3 agents are already running");
    expect(screen.getByLabelText("Task")).toHaveValue("tidy");
  });
});

describe("agent detail", () => {
  const events = [
    ev(1, { kind: "user", text: "tidy the notes" }),
    ev(2, { kind: "text", text: "On it." }),
    ev(3, { kind: "tool-use", tool: "Bash", toolUseId: "t1", input: { command: "git status" } }),
    ev(4, { kind: "tool-result", toolUseId: "t1", output: "clean", isError: false }),
    ev(5, { kind: "approval", approvalId: "p1", tool: "Bash", outcome: "requested" }),
    ev(6, { kind: "result", ok: true, text: "Done", turns: 1, costUsd: 0.02 }),
  ];

  async function expand(name: string) {
    await userEvent.click(await screen.findByRole("button", { name: new RegExp(name) }));
  }

  it("expands to a log of what the agent said and did, as text", async () => {
    mockApi({ "GET /api/agents": list([agent("fix-lcn", "idle", { lastSeq: 6 })]), "GET /api/agents/fix-lcn/events?after=0&limit=200": log(events) });
    render(<Agents />);
    await expand("fix-lcn");
    const region = await screen.findByRole("log", { name: /fix-lcn/ });
    expect(within(region).getByText("tidy the notes")).toBeInTheDocument();
    expect(within(region).getByText("On it.")).toBeInTheDocument();
    expect(within(region).getByText("git status")).toHaveClass("mono");
    expect(within(region).getByText("clean")).toBeInTheDocument();
    expect(within(region).getByText(/approval requested/i)).toBeInTheDocument();
  });

  it("never renders model output as HTML", async () => {
    const hostile = '<img src=x onerror="alert(1)">';
    mockApi({ "GET /api/agents": list([agent("fix-lcn", "idle", { lastSeq: 1 })]), "GET /api/agents/fix-lcn/events?after=0&limit=200": log([ev(1, { kind: "text", text: hostile })]) });
    const { container } = render(<Agents />);
    await expand("fix-lcn");
    expect(await screen.findByText(hostile)).toBeInTheDocument();
    expect(container.querySelector("img")).toBeNull();
  });

  it("fetches only new entries, after the cursor, when the server announces a change", async () => {
    let all = events.slice(0, 2);
    const api = mockApi({
      "GET /api/agents": () => list([agent("fix-lcn", "running", { lastSeq: all.length })]),
      "GET /api/agents/fix-lcn/events?after=0&limit=200": () => log(all),
      "GET /api/agents/fix-lcn/events?after=2&limit=200": () => log(events.slice(2, 4), 4),
    });
    render(<Agents />);
    await expand("fix-lcn");
    await screen.findByText("On it.");
    all = events.slice(0, 4);
    act(() => FakeEventSource.emit({ type: "invalidate", resource: "agents" }));
    expect(await screen.findByText("clean")).toBeInTheDocument();
    expect(screen.getAllByText("On it.")).toHaveLength(1);
    expect(api.count("GET", "/api/agents/fix-lcn/events?after=2&limit=200")).toBeGreaterThan(0);
  });

  it("sends a message and clears the box", async () => {
    const api = mockApi({
      "GET /api/agents": list([agent("fix-lcn", "idle")]),
      "GET /api/agents/fix-lcn/events?after=0&limit=200": log([]),
      "POST /api/agents/fix-lcn/message": { body: { ok: true } },
    });
    render(<Agents />);
    await expand("fix-lcn");
    const box = await screen.findByLabelText(/Message to fix-lcn/);
    await userEvent.type(box, "also fix the tests");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));
    await waitFor(() => expect(api.calls.find((c) => c.method === "POST")?.body).toEqual({ text: "also fix the tests" }));
    await waitFor(() => expect(box).toHaveValue(""));
  });

  it("does not offer a message box for a stopped agent but offers Resume", async () => {
    const api = mockApi({
      "GET /api/agents": list([agent("fix-lcn", "stopped")]),
      "GET /api/agents/fix-lcn/events?after=0&limit=200": log([]),
      "POST /api/agents/fix-lcn/resume": { body: { ok: true } },
    });
    render(<Agents />);
    await expand("fix-lcn");
    await screen.findByRole("log");
    expect(screen.queryByLabelText(/Message to/)).toBeNull();
    await userEvent.click(screen.getByRole("button", { name: "Resume" }));
    await waitFor(() => expect(api.count("POST", "/api/agents/fix-lcn/resume")).toBe(1));
  });

  it("Stop posts to the stop endpoint for a running agent", async () => {
    const api = mockApi({
      "GET /api/agents": list([agent("fix-lcn", "running")]),
      "GET /api/agents/fix-lcn/events?after=0&limit=200": log([]),
      "POST /api/agents/fix-lcn/stop": { body: { ok: true } },
    });
    render(<Agents />);
    await expand("fix-lcn");
    await userEvent.click(await screen.findByRole("button", { name: "Stop" }));
    await waitFor(() => expect(api.count("POST", "/api/agents/fix-lcn/stop")).toBe(1));
  });

  it("shows a failed action's message instead of swallowing it", async () => {
    mockApi({
      "GET /api/agents": list([agent("fix-lcn", "running")]),
      "GET /api/agents/fix-lcn/events?after=0&limit=200": log([]),
      "POST /api/agents/fix-lcn/stop": { status: 500, body: { error: "internal error" } },
    });
    render(<Agents />);
    await expand("fix-lcn");
    await userEvent.click(await screen.findByRole("button", { name: "Stop" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("internal error");
  });

  it("the row button reports whether the log is open", async () => {
    mockApi({ "GET /api/agents": list([agent("fix-lcn")]), "GET /api/agents/fix-lcn/events?after=0&limit=200": log([]) });
    render(<Agents />);
    const btn = await screen.findByRole("button", { name: /fix-lcn/ });
    expect(btn).toHaveAttribute("aria-expanded", "false");
    await userEvent.click(btn);
    expect(btn).toHaveAttribute("aria-expanded", "true");
  });
});
