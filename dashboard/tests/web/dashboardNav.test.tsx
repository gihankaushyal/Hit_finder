import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { DashboardNav } from "../../web/src/components/DashboardNav";
import { mockApi, stubEventSource } from "./helpers";

const routes = {
  "GET /api/repo": { body: { branch: "phase-05", commit: "610ae0e" } },
  "GET /api/ci": { body: { runs: [{ id: 1, status: "completed", conclusion: "success", branch: "main", event: "push", createdAt: "2026-10-01T10:00:00Z", url: "https://x/1" }] } },
  "GET /api/tests": { body: { status: "done", summary: { tests: 3, passed: 3, failed: 0, errors: 0, skipped: 0, durationSec: 1, failures: [] } } },
  "GET /api/approvals": { body: { approvals: [] } },
};

beforeEach(() => {
  stubEventSource();
});
afterEach(() => vi.unstubAllGlobals());

function mount(props: Partial<React.ComponentProps<typeof DashboardNav>> = {}) {
  const onTab = vi.fn();
  const onToggleAgents = vi.fn();
  render(<DashboardNav tab="overview" onTab={onTab} agentsOpen={false} onToggleAgents={onToggleAgents} {...props} />);
  return { onTab, onToggleAgents };
}

describe("DashboardNav", () => {
  it("shows the brand, branch, commit and the CI and test badges", async () => {
    mockApi(routes);
    mount();
    expect(screen.getByRole("heading", { level: 1, name: "Hit_finder" })).toBeInTheDocument();
    expect(await screen.findByLabelText("Current branch")).toHaveTextContent("phase-05");
    expect(screen.getByText("610ae0e")).toBeInTheDocument();
    expect(await screen.findByText("CI Passing")).toBeInTheDocument();
    expect(await screen.findByText("Tests Passing")).toBeInTheDocument();
  });

  it("lists the three tabs as links and marks the current one", () => {
    mockApi(routes);
    mount({ tab: "kanban" });
    const nav = screen.getByRole("navigation", { name: "Dashboard sections" });
    const links = Array.from(nav.querySelectorAll("a"));
    expect(links.map((a) => a.textContent)).toEqual(["Overview", "Kanban Board", "Runs & Checkpoints"]);
    expect(links.map((a) => a.getAttribute("href"))).toEqual(["?", "?tab=kanban", "?tab=runs"]);
    expect(screen.getByRole("link", { name: "Kanban Board" })).toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("link", { name: "Overview" })).not.toHaveAttribute("aria-current");
  });

  it("a plain click switches tab; a modified click is left to the browser", async () => {
    mockApi(routes);
    const { onTab } = mount();
    await userEvent.click(screen.getByRole("link", { name: "Runs & Checkpoints" }));
    expect(onTab).toHaveBeenCalledWith("runs");
    onTab.mockClear();
    const user = userEvent.setup();
    await user.keyboard("{Control>}");
    await user.click(screen.getByRole("link", { name: "Kanban Board" }));
    await user.keyboard("{/Control}");
    expect(onTab).not.toHaveBeenCalled();
  });

  it("copies the commit hash and says so", async () => {
    mockApi(routes);
    mount();
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", { value: { writeText }, configurable: true });
    await userEvent.click(await screen.findByRole("button", { name: "Copy commit hash" }));
    expect(writeText).toHaveBeenCalledWith("610ae0e");
    await waitFor(() => expect(document.querySelector(".nav__commit [role=status]")).toHaveTextContent("Copied"));
  });

  it("the Agents button reflects and toggles the drawer", async () => {
    mockApi(routes);
    const { onToggleAgents } = mount({ agentsOpen: true });
    const btn = screen.getByRole("button", { name: "Agents" });
    expect(btn).toHaveAttribute("aria-expanded", "true");
    await userEvent.click(btn);
    expect(onToggleAgents).toHaveBeenCalledTimes(1);
  });

  it("marks a failing or still-running state with its word, not just a colour", async () => {
    mockApi({ ...routes, "GET /api/ci": { body: { runs: [{ id: 2, status: "completed", conclusion: "failure", branch: "x", event: "push", createdAt: "2026-10-01T10:00:00Z", url: "https://x/2" }] } } });
    mount();
    expect(await screen.findByText("CI Failing")).toBeInTheDocument();
  });
});
