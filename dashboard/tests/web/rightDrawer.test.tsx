import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { RightDrawer } from "../../web/src/components/RightDrawer";
import { mockApi, stubEventSource } from "./helpers";

const routes = {
  "GET /api/approvals": { body: { approvals: [] } },
  "GET /api/agents": { body: { agents: [] } },
  "GET /api/skills": { body: { skills: [] } },
  "GET /api/chief": { body: { status: "idle", sessionId: null } },
  "GET /api/chief/events": { body: { events: [], next: 0 } },
};

beforeEach(() => {
  stubEventSource();
  mockApi(routes);
});
afterEach(() => vi.unstubAllGlobals());

describe("RightDrawer", () => {
  it("is inert while closed (no focus, not read out) and live while open", () => {
    const { rerender } = render(<RightDrawer open={false} onClose={() => {}} />);
    const drawer = () => document.getElementById("agents-drawer")!;
    expect(drawer()).toHaveAttribute("inert");
    expect(drawer()).toHaveAttribute("data-open", "false");
    rerender(<RightDrawer open onClose={() => {}} />);
    expect(drawer()).not.toHaveAttribute("inert");
    expect(screen.getByRole("dialog", { name: "Agents and chief of staff" })).toBeInTheDocument();
  });

  it("moves focus to the close button when it opens", () => {
    const { rerender } = render(<RightDrawer open={false} onClose={() => {}} />);
    rerender(<RightDrawer open onClose={() => {}} />);
    expect(screen.getAllByRole("button", { name: "Close agents panel" }).find((b) => b === document.activeElement)).toBeTruthy();
  });

  it("keeps the mouse-only backdrop out of the accessibility tree, leaving one close button", () => {
    render(<RightDrawer open onClose={() => {}} />);
    expect(document.querySelector(".scrim")).toHaveAttribute("aria-hidden", "true");
    expect(screen.getAllByRole("button", { name: "Close agents panel" })).toHaveLength(1);
  });

  it("closes on Escape, on the close button and on the backdrop", async () => {
    const onClose = vi.fn();
    render(<RightDrawer open onClose={onClose} />);
    await userEvent.keyboard("{Escape}");
    expect(onClose).toHaveBeenCalledTimes(1);
    await userEvent.click(screen.getByRole("dialog").querySelector("button[aria-label='Close agents panel']")!);
    expect(onClose).toHaveBeenCalledTimes(2);
    await userEvent.click(document.querySelector(".scrim")!);
    expect(onClose).toHaveBeenCalledTimes(3);
  });

  it("ignores Escape while closed", async () => {
    const onClose = vi.fn();
    render(<RightDrawer open={false} onClose={onClose} />);
    await userEvent.keyboard("{Escape}");
    expect(onClose).not.toHaveBeenCalled();
  });

  it("is modal: Tab wraps inside the drawer in both directions", async () => {
    render(<RightDrawer open onClose={() => {}} />);
    const drawer = document.getElementById("agents-drawer")!;
    expect(drawer).toHaveAttribute("aria-modal", "true");
    const stops = Array.from(drawer.querySelectorAll<HTMLElement>("button:not([disabled]), input, select, textarea, a[href]"));
    const last = stops[stops.length - 1];
    last.focus();
    await userEvent.tab();
    expect(document.activeElement).toBe(stops[0]);
    await userEvent.tab({ shift: true });
    expect(document.activeElement).toBe(last);
  });

  it("does not close when Escape is pressed inside a select", async () => {
    const onClose = vi.fn();
    render(<RightDrawer open onClose={onClose} />);
    const select = document.createElement("select");
    document.getElementById("agents-drawer")!.appendChild(select);
    select.focus();
    await userEvent.keyboard("{Escape}");
    expect(onClose).not.toHaveBeenCalled();
  });
});
