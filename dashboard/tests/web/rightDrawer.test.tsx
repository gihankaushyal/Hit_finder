import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { RightDrawer } from "../../web/src/components/RightDrawer";
import { mockApi, stubEventSource } from "./helpers";

const routes = {
  "GET /api/approvals": { body: { approvals: [] } },
  "GET /api/agents": { body: { agents: [] } },
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
});
