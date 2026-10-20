import { describe, it, expect, afterEach, beforeEach, vi } from "vitest";
import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Approvals } from "../../web/src/panels/Approvals";
import { TopBar } from "../../web/src/panels/TopBar";
import { FakeEventSource, mockApi, stubEventSource } from "./helpers";
import type { Approval } from "../../shared/types";

beforeEach(() => stubEventSource());
afterEach(() => vi.unstubAllGlobals());

const card = (id: string, over: Partial<Approval> = {}): Approval => ({
  id, agentId: "fix-lcn", agentName: "fix-lcn", tool: "Bash", input: { command: "git push origin HEAD" }, summary: "git push origin HEAD",
  reason: null, requestedAt: "2026-10-20T08:00:00.000Z", ...over,
});
const none = { body: { approvals: [] } };

describe("Approvals", () => {
  it("shows an empty state", async () => {
    mockApi({ "GET /api/approvals": none });
    render(<Approvals />);
    expect(await screen.findByText(/Nothing is waiting/)).toBeInTheDocument();
  });

  it("shows the agent, the tool and the exact command on each card", async () => {
    mockApi({ "GET /api/approvals": { body: { approvals: [card("a1")] } } });
    render(<Approvals />);
    const article = await screen.findByRole("article", { name: /fix-lcn/ });
    expect(within(article).getByText("Bash")).toBeInTheDocument();
    expect(within(article).getByText("git push origin HEAD")).toHaveClass("mono");
    expect(within(article).getByRole("button", { name: /Approve/ })).toBeInTheDocument();
    expect(within(article).getByRole("button", { name: /Deny/ })).toBeInTheDocument();
  });

  it("shows a long command in full, never truncated", async () => {
    const long = `echo ${"x".repeat(2000)}`;
    mockApi({ "GET /api/approvals": { body: { approvals: [card("a1", { input: { command: long }, summary: "echo..." })] } } });
    render(<Approvals />);
    const code = await screen.findByText(long);
    expect(code.textContent).toHaveLength(long.length);
  });

  it("renders model-supplied text as text, never as HTML", async () => {
    const hostile = '<img src=x onerror="alert(1)"><script>alert(2)</script>';
    mockApi({ "GET /api/approvals": { body: { approvals: [card("a1", { input: { command: hostile }, agentName: "<b>x</b>" })] } } });
    const { container } = render(<Approvals />);
    expect(await screen.findByText(hostile)).toBeInTheDocument();
    expect(container.querySelector("img, script, b")).toBeNull();
  });

  it("shows the file and content of an edit, and falls back to the raw input for other tools", async () => {
    mockApi({
      "GET /api/approvals": {
        body: { approvals: [card("e1", { tool: "Edit", input: { file_path: "/w/a.ts", old_string: "one", new_string: "two" } }), card("m1", { tool: "mcp__x__go", input: { a: 1 } })] },
      },
    });
    render(<Approvals />);
    const edit = await screen.findByRole("article", { name: /Edit/ });
    expect(within(edit).getByText("/w/a.ts")).toBeInTheDocument();
    expect(within(edit).getByText(/"new_string": "two"/)).toBeInTheDocument();
    expect(within(screen.getByRole("article", { name: /mcp__x__go/ })).getByText(/"a": 1/)).toBeInTheDocument();
  });

  it("Approve posts allow and the card disappears when the server drops it", async () => {
    let pending = [card("a1")];
    const api = mockApi({
      "GET /api/approvals": () => ({ body: { approvals: pending } }),
      "POST /api/approvals/a1/decision": () => ((pending = []), { body: { ok: true } }),
    });
    render(<Approvals />);
    await userEvent.click(await screen.findByRole("button", { name: /Approve/ }));
    await waitFor(() => expect(api.calls.find((c) => c.method === "POST")?.body).toEqual({ decision: "allow" }));
    await waitFor(() => expect(screen.queryByRole("article")).toBeNull());
    expect(screen.getByText(/Nothing is waiting/)).toBeInTheDocument();
  });

  it("Deny posts deny", async () => {
    const api = mockApi({ "GET /api/approvals": { body: { approvals: [card("a1")] } }, "POST /api/approvals/a1/decision": { body: { ok: true } } });
    render(<Approvals />);
    await userEvent.click(await screen.findByRole("button", { name: /Deny/ }));
    await waitFor(() => expect(api.calls.find((c) => c.method === "POST")?.body).toEqual({ decision: "deny" }));
  });

  it("sends one decision even if clicked twice, and disables both buttons meanwhile", async () => {
    let release!: () => void;
    const gate = new Promise<void>((r) => (release = r));
    const api = mockApi({
      "GET /api/approvals": { body: { approvals: [card("a1")] } },
      "POST /api/approvals/a1/decision": async () => (await gate, { body: { ok: true } }),
    });
    render(<Approvals />);
    const approve = await screen.findByRole("button", { name: /Approve/ });
    await userEvent.dblClick(approve);
    expect(approve).toBeDisabled();
    expect(screen.getByRole("button", { name: /Deny/ })).toBeDisabled();
    release();
    await waitFor(() => expect(api.count("POST", "/api/approvals/a1/decision")).toBe(1));
  });

  it("keeps the card and says so when the decision fails", async () => {
    mockApi({ "GET /api/approvals": { body: { approvals: [card("a1")] } }, "POST /api/approvals/a1/decision": { status: 404, body: { error: "approval not found" } } });
    render(<Approvals />);
    await userEvent.click(await screen.findByRole("button", { name: /Approve/ }));
    expect(await screen.findByRole("alert")).toHaveTextContent("approval not found");
    expect(screen.getByRole("article")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Approve/ })).toBeEnabled();
  });

  it("refetches when the server announces new approvals", async () => {
    let list: Approval[] = [];
    const api = mockApi({ "GET /api/approvals": () => ({ body: { approvals: list } }) });
    render(<Approvals />);
    await screen.findByText(/Nothing is waiting/);
    list = [card("a1")];
    act(() => FakeEventSource.emit({ type: "invalidate", resource: "approvals" }));
    expect(await screen.findByRole("article")).toBeInTheDocument();
    expect(api.count("GET", "/api/approvals")).toBeGreaterThan(1);
  });

  it("announces the number waiting through a polite live region", async () => {
    mockApi({ "GET /api/approvals": { body: { approvals: [card("a1"), card("a2")] } } });
    render(<Approvals />);
    await screen.findAllByRole("article");
    const live = screen.getByRole("status");
    expect(live).toHaveAttribute("aria-live", "polite");
    expect(live).toHaveTextContent("2 approvals waiting");
  });

  it("moves focus to the next card after a decision", async () => {
    let list = [card("a1"), card("a2", { agentName: "other" })];
    mockApi({ "GET /api/approvals": () => ({ body: { approvals: list } }), "POST /api/approvals/a1/decision": () => ((list = list.slice(1)), { body: { ok: true } }) });
    render(<Approvals />);
    const first = (await screen.findAllByRole("button", { name: /Approve/ }))[0];
    await userEvent.click(first);
    await waitFor(() => expect(screen.getAllByRole("article")).toHaveLength(1));
    await waitFor(() => expect(screen.getByRole("button", { name: /Approve/ })).toHaveFocus());
  });
});

describe("top bar counter", () => {
  const routes = { "GET /api/repo": { body: { branch: "phase-05", commit: "abc" } }, "GET /api/ci": { body: { runs: [] } }, "GET /api/tests": { body: { status: "idle" } } };

  it("shows how many approvals are waiting, as a link to the cards", async () => {
    mockApi({ ...routes, "GET /api/approvals": { body: { approvals: [card("a1"), card("a2")] } } });
    render(<TopBar />);
    const link = await screen.findByRole("link", { name: /2 waiting for approval/ });
    expect(link).toHaveAttribute("href", "#approvals");
  });

  it("is absent when nothing is waiting", async () => {
    mockApi({ ...routes, "GET /api/approvals": none });
    render(<TopBar />);
    await screen.findByText("phase-05");
    expect(screen.queryByRole("link", { name: /waiting for approval/ })).toBeNull();
  });
});
