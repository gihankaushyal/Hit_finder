import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { App } from "../../web/src/App";
import { mockApi, stubEventSource } from "./helpers";

// The terminal pulls in xterm, which jsdom cannot run; the drawer is covered by terminal.test.tsx.
vi.mock("../../web/src/panels/TerminalDrawer", () => ({ TerminalDrawer: () => <footer>Terminal</footer> }));

const empty = { body: {} };
const routes = {
  "GET /api/health": { body: { ok: true } },
  "GET /api/repo": { body: { branch: "phase-05", commit: "abc1234" } },
  "GET /api/ci": { body: { runs: [] } },
  "GET /api/tests": { body: { status: "idle" } },
  "GET /api/approvals": { body: { approvals: [] } },
  "GET /api/agents": { body: { agents: [] } },
  "GET /api/prs": { body: { latest: null, recent: [], open: [] } },
  "GET /api/issues": { body: [] },
  "GET /api/kanban": { body: { columns: { todo: [], "in-progress": [], blocked: [], done: [] }, conflicts: [], conflictItems: [], lastSyncAt: null, syncError: null, imported: false } },
  "GET /api/runs": { body: { runs: [], scannedAt: "2026-10-01T00:00:00Z", checkpointsDirExists: true, truncated: false, error: null } },
  "GET /api/chief": empty,
};

beforeEach(() => {
  stubEventSource();
  mockApi(routes);
});
afterEach(() => vi.unstubAllGlobals());

const region = (name: string) => screen.queryByRole("region", { name });

describe("App tabs", () => {
  it("opens on Overview with the PR, tests, kanban and runs panels together", async () => {
    render(<App />);
    expect(await screen.findByRole("region", { name: "Latest pull request" })).toBeInTheDocument();
    expect(region("Tests")).toBeInTheDocument();
    expect(region("Kanban")).toBeInTheDocument();
    expect(region("Active Runs")).toBeInTheDocument();
    expect(region("Issues")).toBeNull();
  });

  it("Kanban Board puts the board and the issues side by side, nothing else", async () => {
    window.history.replaceState(null, "", "/?tab=kanban");
    render(<App />);
    expect(await screen.findByRole("region", { name: "Kanban" })).toBeInTheDocument();
    expect(region("Issues")).toBeInTheDocument();
    expect(region("Latest pull request")).toBeNull();
    expect(region("Active Runs")).toBeNull();
  });

  it("Runs & Checkpoints shows only the runs table", async () => {
    window.history.replaceState(null, "", "/?tab=runs");
    render(<App />);
    expect(await screen.findByRole("region", { name: "Active Runs" })).toBeInTheDocument();
    expect(region("Kanban")).toBeNull();
  });

  it("an unknown tab falls back to Overview", async () => {
    window.history.replaceState(null, "", "/?tab=nope");
    render(<App />);
    expect(await screen.findByRole("region", { name: "Latest pull request" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Overview" })).toHaveAttribute("aria-current", "page");
  });

  it("clicking a tab changes the view and the address, and Back returns", async () => {
    const user = userEvent.setup();
    render(<App />);
    await screen.findByRole("region", { name: "Latest pull request" });
    await user.click(screen.getByRole("link", { name: "Runs & Checkpoints" }));
    expect(await screen.findByRole("region", { name: "Active Runs" })).toBeInTheDocument();
    expect(region("Latest pull request")).toBeNull();
    expect(window.location.search).toBe("?tab=runs");
    await user.click(screen.getByRole("link", { name: "Overview" }));
    expect(await screen.findByRole("region", { name: "Latest pull request" })).toBeInTheDocument();
    expect(window.location.search).toBe("");
  });

  it("the Agents button opens the drawer and Escape closes it, returning focus to the button", async () => {
    const user = userEvent.setup();
    render(<App />);
    const button = await screen.findByRole("button", { name: "Agents" });
    await user.click(button);
    expect(button).toHaveAttribute("aria-expanded", "true");
    expect(document.getElementById("agents-drawer")).not.toHaveAttribute("inert");
    await user.keyboard("{Escape}");
    expect(button).toHaveAttribute("aria-expanded", "false");
    expect(document.getElementById("agents-drawer")).toHaveAttribute("inert");
    expect(button).toHaveFocus();
  });
});
