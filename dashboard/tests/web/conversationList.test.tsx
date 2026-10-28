import { describe, it, expect, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ConversationList, type ConversationListProps } from "../../web/src/components/ConversationList";
import type { ConversationInfo } from "../../shared/types";

const row = (over: Partial<ConversationInfo>): ConversationInfo => ({
  id: "c-a", title: "Alpha", createdAt: "2026-10-20T08:00:00.000Z", updatedAt: "2026-10-20T09:00:00.000Z", turns: 2, costUsd: 0.5,
  archived: false, readOnly: null, active: false, seededFrom: null, ...over,
});

function setup(rows: ConversationInfo[], over: Partial<ConversationListProps> = {}) {
  const fns = { onView: vi.fn(), onResume: vi.fn(), onRename: vi.fn(), onArchive: vi.fn() };
  render(<ConversationList rows={rows} viewingId={null} busy={false} {...fns} {...over} />);
  return fns;
}

describe("ConversationList", () => {
  it("shows title, turns, cost and which one is current", () => {
    setup([row({ id: "c-now", title: "Now", active: true, turns: 1, costUsd: null }), row({})]);
    const now = screen.getByText("Now").closest("li") as HTMLElement;
    expect(within(now).getByText("Current")).toBeInTheDocument();
    expect(within(now).getByText(/1 turn\b/)).toBeInTheDocument();
    const alpha = screen.getByText("Alpha").closest("li") as HTMLElement;
    expect(within(alpha).getByText(/2 turns/)).toBeInTheDocument();
    expect(within(alpha).getByText(/about \$0\.50/)).toBeInTheDocument();
  });

  it("reads and resumes a parked conversation, but not the current one", async () => {
    const fns = setup([row({ id: "c-now", title: "Now", active: true }), row({})]);
    expect(screen.queryByRole("button", { name: "Read Now" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Resume Now" })).toBeNull();
    await userEvent.click(screen.getByRole("button", { name: "Read Alpha" }));
    expect(fns.onView).toHaveBeenCalledWith("c-a");
    await userEvent.click(screen.getByRole("button", { name: "Resume Alpha" }));
    expect(fns.onResume).toHaveBeenCalledWith("c-a");
  });

  it("offers no Resume for a read-only conversation, and marks it", () => {
    setup([row({ readOnly: "expired" })]);
    expect(screen.getByText("Read-only")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Read Alpha" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Resume Alpha" })).toBeNull();
  });

  it("renames with Enter, cancels with Escape, and ignores an empty title", async () => {
    const fns = setup([row({})]);
    await userEvent.click(screen.getByRole("button", { name: "Rename Alpha" }));
    const box = screen.getByLabelText("Conversation title");
    await userEvent.clear(box);
    await userEvent.type(box, "Beta{Enter}");
    expect(fns.onRename).toHaveBeenCalledWith("c-a", "Beta");
    await userEvent.click(screen.getByRole("button", { name: "Rename Alpha" }));
    await userEvent.type(screen.getByLabelText("Conversation title"), "{Escape}");
    expect(screen.queryByLabelText("Conversation title")).toBeNull();
    await userEvent.click(screen.getByRole("button", { name: "Rename Alpha" }));
    await userEvent.clear(screen.getByLabelText("Conversation title"));
    await userEvent.type(screen.getByLabelText("Conversation title"), "{Enter}");
    expect(fns.onRename).toHaveBeenCalledTimes(1);
  });

  it("hides archived conversations until asked, and restores one", async () => {
    const fns = setup([row({}), row({ id: "c-b", title: "Old", archived: true })]);
    expect(screen.queryByText("Old")).toBeNull();
    await userEvent.click(screen.getByRole("button", { name: "Archive Alpha" }));
    expect(fns.onArchive).toHaveBeenCalledWith("c-a", true);
    await userEvent.click(screen.getByLabelText("Show archived"));
    expect(screen.getByText("Old")).toBeInTheDocument();
    expect(screen.queryByText("Alpha")).toBeNull();
    await userEvent.click(screen.getByRole("button", { name: "Restore Old" }));
    expect(fns.onArchive).toHaveBeenCalledWith("c-b", false);
  });

  it("calls an untitled conversation 'Untitled' and disables actions while busy", () => {
    setup([row({ title: null })], { busy: true });
    expect(screen.getByRole("button", { name: "Read Untitled" })).toBeDisabled();
  });
});
