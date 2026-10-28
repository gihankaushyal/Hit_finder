import { describe, it, expect, afterEach, beforeEach, vi } from "vitest";
import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ChiefChat } from "../../web/src/panels/ChiefChat";
import { FakeEventSource, mockApi, stubEventSource } from "./helpers";
import type { AgentEvent, AgentEventBody, AgentEventsResponse, ChiefSummary, ConversationInfo, ConversationsResponse } from "../../shared/types";

beforeEach(() => stubEventSource());
afterEach(() => vi.unstubAllGlobals());

const chief = (over: Partial<ChiefSummary> = {}): { body: ChiefSummary } => ({
  body: { state: "idle", model: "opus", turns: 0, costUsd: null, sessionId: null, error: null, pendingApprovals: 0, lastSeq: 0, conversationId: "c-now", title: null, readOnly: null, ...over },
});
const ev = (seq: number, e: AgentEventBody): AgentEvent => ({ ...e, seq, at: "2026-10-20T08:01:00.000Z" });
const log = (events: AgentEvent[]): { body: AgentEventsResponse } => ({ body: { events, lastSeq: events.length, conversationId: "c-now" } });
const EVENTS = "GET /api/chief/events?after=0&limit=200";

describe("ChiefChat", () => {
  it("shows a hint and a composer when the conversation is empty", async () => {
    mockApi({ "GET /api/chief": chief(), [EVENTS]: log([]) });
    render(<ChiefChat />);
    expect(await screen.findByText(/Ask the chief of staff/)).toBeInTheDocument();
    expect(screen.getByLabelText("Message to the chief of staff")).toBeInTheDocument();
    expect(screen.getByText("Idle")).toBeInTheDocument();
  });

  it("sends a message, clears the box, and Ctrl+Enter sends too", async () => {
    const api = mockApi({ "GET /api/chief": chief(), [EVENTS]: log([]), "POST /api/chief/message": { body: { ok: true } } });
    render(<ChiefChat />);
    const box = await screen.findByLabelText("Message to the chief of staff");
    await userEvent.type(box, "start a worker for the docs");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));
    await waitFor(() => expect(api.calls.find((c) => c.method === "POST")?.body).toEqual({ text: "start a worker for the docs" }));
    await waitFor(() => expect(box).toHaveValue(""));
    await userEvent.type(box, "second{Control>}{Enter}{/Control}");
    await waitFor(() => expect(api.calls.filter((c) => c.method === "POST")).toHaveLength(2));
  });

  it("will not send an empty message", async () => {
    const api = mockApi({ "GET /api/chief": chief(), [EVENTS]: log([]) });
    render(<ChiefChat />);
    expect(await screen.findByRole("button", { name: "Send" })).toBeDisabled();
    expect(api.count("POST", "/api/chief/message")).toBe(0);
  });

  it("shows the conversation: you, the chief, and each tool call with its arguments", async () => {
    mockApi({
      "GET /api/chief": chief({ turns: 1, lastSeq: 5 }),
      [EVENTS]: log([
        ev(1, { kind: "user", text: "spawn a docs worker" }),
        ev(2, { kind: "text", text: "Starting one." }),
        ev(3, { kind: "tool-use", tool: "mcp__dashboard__spawn_worker", toolUseId: "t1", input: { name: "docs-one", task: "write docs" } }),
        ev(4, { kind: "tool-result", toolUseId: "t1", output: "Started worker docs-one on branch phase-05-agent-docs-one.", isError: false }),
        ev(5, { kind: "result", ok: true, text: "", turns: 1, costUsd: 0.1 }),
      ]),
    });
    render(<ChiefChat />);
    const region = await screen.findByRole("log", { name: "Conversation with the chief of staff" });
    expect(within(region).getByText("spawn a docs worker")).toBeInTheDocument();
    expect(within(region).getByText("Starting one.")).toBeInTheDocument();
    expect(within(region).getByText("spawn_worker")).toBeInTheDocument();
    expect(within(region).getByText(/"name": "docs-one"/)).toBeInTheDocument();
    // A result is collapsed to a one-line preview in its summary, with the full text in the body below it.
    const result = within(region).getByText("Result").closest("details") as HTMLElement;
    expect(within(result).getAllByText(/Started worker docs-one/)).toHaveLength(2);
  });

  it("renders everything as text, never HTML", async () => {
    const hostile = '<img src=x onerror="alert(1)">';
    mockApi({ "GET /api/chief": chief({ lastSeq: 1 }), [EVENTS]: log([ev(1, { kind: "text", text: hostile })]) });
    const { container } = render(<ChiefChat />);
    expect(await screen.findByText(hostile)).toBeInTheDocument();
    expect(container.querySelector("img")).toBeNull();
  });

  it("links to the approval cards while a tool call is waiting", async () => {
    mockApi({ "GET /api/chief": chief({ state: "waiting-approval", pendingApprovals: 1 }), [EVENTS]: log([]) });
    render(<ChiefChat />);
    expect(await screen.findByText("Waiting for approval")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /1 pending/ })).toHaveAttribute("href", "#approvals");
  });

  it("Stop appears while it is working and posts to stop", async () => {
    const api = mockApi({ "GET /api/chief": chief({ state: "running" }), [EVENTS]: log([]), "POST /api/chief/stop": { body: { ok: true } } });
    render(<ChiefChat />);
    await userEvent.click(await screen.findByRole("button", { name: "Stop" }));
    await waitFor(() => expect(api.count("POST", "/api/chief/stop")).toBe(1));
  });

  it("starting a new conversation asks first, then posts reset", async () => {
    const api = mockApi({ "GET /api/chief": chief({ turns: 2 }), [EVENTS]: log([]), "POST /api/chief/reset": { body: { ok: true } } });
    render(<ChiefChat />);
    await userEvent.click(await screen.findByRole("button", { name: "New conversation" }));
    expect(api.count("POST", "/api/chief/reset")).toBe(0);
    await userEvent.click(screen.getByRole("button", { name: "Yes, start over" }));
    await waitFor(() => expect(api.count("POST", "/api/chief/reset")).toBe(1));
  });

  it("lets you start over when a saved session exists even if no turn finished, or after a failure", async () => {
    mockApi({ "GET /api/chief": chief({ turns: 0, sessionId: "s-1" }), [EVENTS]: log([]) });
    const { unmount } = render(<ChiefChat />);
    expect(await screen.findByRole("button", { name: "New conversation" })).toBeEnabled();
    unmount();
    mockApi({ "GET /api/chief": chief({ turns: 0, state: "failed", error: "x" }), [EVENTS]: log([]) });
    render(<ChiefChat />);
    expect(await screen.findByRole("button", { name: "New conversation" })).toBeEnabled();
  });

  it("starts reading again from the top when the server's log was replaced by a new conversation", async () => {
    let events = [ev(1, { kind: "user", text: "old" }), ev(2, { kind: "text", text: "old reply" }), ev(3, { kind: "text", text: "older still" })];
    mockApi({
      "GET /api/chief": () => chief({ lastSeq: events.length }),
      [EVENTS]: () => log(events),
      "GET /api/chief/events?after=3&limit=200": () => ({ body: { events: [], lastSeq: events.length } }),
    });
    render(<ChiefChat />);
    await screen.findByText("older still");
    events = [ev(1, { kind: "state", state: "idle", detail: "New conversation" }), ev(2, { kind: "user", text: "fresh" })];
    act(() => FakeEventSource.emit({ type: "invalidate", resource: "chat" }));
    expect(await screen.findByText("fresh")).toBeInTheDocument();
    expect(screen.queryByText("old reply")).toBeNull();
  });

  it("fetches only new entries, after the cursor, when chat changes", async () => {
    let events = [ev(1, { kind: "user", text: "hi" }), ev(2, { kind: "text", text: "hello" })];
    const api = mockApi({
      "GET /api/chief": () => chief({ lastSeq: events.length }),
      [EVENTS]: () => log(events.slice(0, 2)),
      "GET /api/chief/events?after=2&limit=200": () => ({ body: { events: events.slice(2), lastSeq: events.length } }),
    });
    render(<ChiefChat />);
    await screen.findByText("hello");
    events = [...events, ev(3, { kind: "text", text: "one more thing" })];
    act(() => FakeEventSource.emit({ type: "invalidate", resource: "chat" }));
    expect(await screen.findByText("one more thing")).toBeInTheDocument();
    expect(screen.getAllByText("hello")).toHaveLength(1);
    expect(api.count("GET", "/api/chief/events?after=2&limit=200")).toBeGreaterThan(0);
  });

  it("announces the finished reply, not partial text, through a polite live region", async () => {
    mockApi({ "GET /api/chief": chief({ turns: 1, lastSeq: 2 }), [EVENTS]: log([ev(1, { kind: "user", text: "hi" }), ev(2, { kind: "text", text: "All done." })]) });
    render(<ChiefChat />);
    await screen.findByText("All done.");
    const live = screen.getByRole("status");
    expect(live).toHaveAttribute("aria-live", "polite");
    await waitFor(() => expect(live).toHaveTextContent("Chief of staff: All done."));
  });

  it("shows a failed send and keeps the text", async () => {
    mockApi({ "GET /api/chief": chief(), [EVENTS]: log([]), "POST /api/chief/message": { status: 400, body: { error: "The message cannot be empty." } } });
    render(<ChiefChat />);
    const box = await screen.findByLabelText("Message to the chief of staff");
    await userEvent.type(box, "x");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("The message cannot be empty.");
    expect(box).toHaveValue("x");
  });

  it("shows the session error but still lets you write", async () => {
    mockApi({ "GET /api/chief": chief({ state: "failed", error: "The chief of staff session ended with an error." }), [EVENTS]: log([]) });
    render(<ChiefChat />);
    expect(await screen.findByText("The chief of staff session ended with an error.")).toBeInTheDocument();
    expect(screen.getByLabelText("Message to the chief of staff")).toBeEnabled();
  });

  it("does not announce the old replies of a conversation that has just become the active one", async () => {
    const a = [ev(1, { kind: "user", text: "hi" }), ev(2, { kind: "text", text: "alpha reply" })];
    const b = [ev(1, { kind: "user", text: "yo" }), ev(2, { kind: "text", text: "beta old" })];
    let cur: { conv: string; events: AgentEvent[] } = { conv: "c-a", events: a };
    const page = (after: number) => ({ body: { events: cur.events.filter((e) => e.seq > after), lastSeq: cur.events[cur.events.length - 1].seq, conversationId: cur.conv } as AgentEventsResponse });
    mockApi({
      "GET /api/chief": () => chief({ conversationId: cur.conv, turns: 1 }),
      "GET /api/chief/events?after=0&limit=200": () => page(0),
      "GET /api/chief/events?after=2&limit=200": () => page(2),
      "GET /api/chief/events?after=3&limit=200": () => page(3),
    });
    render(<ChiefChat />);
    const live = await screen.findByRole("status");
    await waitFor(() => expect(live).toHaveTextContent("Chief of staff: alpha reply"));
    cur = { conv: "c-b", events: b };
    act(() => FakeEventSource.emit({ type: "invalidate", resource: "chat" }));
    await screen.findByText("beta old");
    await new Promise((r) => setTimeout(r, 30));
    expect(live).toHaveTextContent("Chief of staff: alpha reply"); // "beta old" is not new
    cur = { conv: "c-b", events: [...b, ev(3, { kind: "text", text: "beta new" })] };
    act(() => FakeEventSource.emit({ type: "invalidate", resource: "chat" }));
    await waitFor(() => expect(live).toHaveTextContent("Chief of staff: beta new"));
  });

  describe("conversations", () => {
    const info = (over: Partial<ConversationInfo>): ConversationInfo => ({
      id: "c-old", title: "Old topic", createdAt: "2026-10-20T08:00:00.000Z", updatedAt: "2026-10-20T09:00:00.000Z", turns: 1, costUsd: null,
      archived: false, readOnly: null, active: false, seededFrom: null, ...over,
    });
    const convs = (rows: ConversationInfo[]): { body: ConversationsResponse } => ({ body: { conversations: rows } });
    const NOW = info({ id: "c-now", title: "Now", active: true });
    const OLD_EVENTS = "GET /api/chief/conversations/c-old/events?after=0&limit=200";
    const base = () => ({ "GET /api/chief": chief({ conversationId: "c-now", title: "Now", turns: 1 }), [EVENTS]: log([]), "GET /api/chief/conversations": convs([NOW, info({})]) });

    it("opens the list, reads a parked conversation, and hides the composer while reading", async () => {
      mockApi({ ...base(), [OLD_EVENTS]: log([ev(1, { kind: "user", text: "what we said before" })]) });
      render(<ChiefChat />);
      await userEvent.click(await screen.findByRole("button", { name: "Conversations" }));
      await userEvent.click(await screen.findByRole("button", { name: "Read Old topic" }));
      expect(await screen.findByText("what we said before")).toBeInTheDocument();
      expect(screen.getByText("You are reading an earlier conversation.")).toBeInTheDocument();
      expect(screen.queryByLabelText("Message to the chief of staff")).toBeNull();
      await userEvent.click(screen.getByRole("button", { name: "Back to current" }));
      expect(await screen.findByLabelText("Message to the chief of staff")).toBeInTheDocument();
    });

    it("resumes the conversation being read", async () => {
      const api = mockApi({ ...base(), [OLD_EVENTS]: log([]), "POST /api/chief/conversations/c-old/resume": { body: { ok: true } } });
      render(<ChiefChat />);
      await userEvent.click(await screen.findByRole("button", { name: "Conversations" }));
      await userEvent.click(await screen.findByRole("button", { name: "Read Old topic" }));
      await userEvent.click(await screen.findByRole("button", { name: "Resume this conversation" }));
      await waitFor(() => expect(api.count("POST", "/api/chief/conversations/c-old/resume")).toBe(1));
      expect(await screen.findByLabelText("Message to the chief of staff")).toBeInTheDocument();
    });

    it("renames and archives through the API", async () => {
      const api = mockApi({ ...base(), "PATCH /api/chief/conversations/c-old": { body: { ok: true } } });
      render(<ChiefChat />);
      await userEvent.click(await screen.findByRole("button", { name: "Conversations" }));
      await userEvent.click(await screen.findByRole("button", { name: "Archive Old topic" }));
      await waitFor(() => expect(api.calls.find((c) => c.method === "PATCH")?.body).toEqual({ archived: true }));
      await userEvent.click(screen.getByRole("button", { name: "Rename Old topic" }));
      await userEvent.type(screen.getByLabelText("Conversation title"), " v2{Enter}");
      await waitFor(() => expect(api.calls.filter((c) => c.method === "PATCH")[1]?.body).toEqual({ title: "Old topic v2" }));
    });

    it("says why a read-only conversation cannot be continued", async () => {
      mockApi({ ...base(), "GET /api/chief/conversations": convs([NOW, info({ readOnly: "legacy" })]), [OLD_EVENTS]: log([]) });
      render(<ChiefChat />);
      await userEvent.click(await screen.findByRole("button", { name: "Conversations" }));
      await userEvent.click(await screen.findByRole("button", { name: "Read Old topic" }));
      expect(await screen.findByText(/can be read but not continued/)).toBeInTheDocument();
      expect(screen.queryByRole("button", { name: "Resume this conversation" })).toBeNull();
    });

    it("will not switch conversations while the chief is working", async () => {
      mockApi({ ...base(), "GET /api/chief": chief({ conversationId: "c-now", state: "running" }), [OLD_EVENTS]: log([]) });
      render(<ChiefChat />);
      await userEvent.click(await screen.findByRole("button", { name: "Conversations" }));
      expect(await screen.findByRole("button", { name: "Resume Old topic" })).toBeDisabled();
      await userEvent.click(screen.getByRole("button", { name: "Read Old topic" }));
      expect(await screen.findByRole("button", { name: "Resume this conversation" })).toBeDisabled();
    });

    it("lists nothing to read when the conversations could not be loaded", async () => {
      mockApi({ "GET /api/chief": chief({ conversationId: "c-now" }), [EVENTS]: log([]), [OLD_EVENTS]: log([]) });
      render(<ChiefChat />);
      await userEvent.click(await screen.findByRole("button", { name: "Conversations" }));
      expect(screen.queryByRole("button", { name: "Read Old topic" })).toBeNull(); // the list could not be loaded
    });

    it("says so when the details of the conversation being read cannot be loaded", async () => {
      mockApi({ "GET /api/chief": chief({ conversationId: "c-now" }), [EVENTS]: log([]), [OLD_EVENTS]: log([]) });
      render(<ChiefChat />);
      await userEvent.click(await screen.findByRole("button", { name: "Conversations" }));
      expect(screen.queryByRole("button", { name: "Resume this conversation" })).toBeNull();
    });
  });
});
