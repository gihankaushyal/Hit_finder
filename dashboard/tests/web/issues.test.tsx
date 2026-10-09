import { describe, it, expect, afterEach, beforeEach, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Issues } from "../../web/src/panels/Issues";
import { deferred, mockApi, stubEventSource } from "./helpers";
import type { Issue } from "../../shared/types";

const issue = (n: number, over: Partial<Issue> = {}): Issue => ({
  number: n, title: `Issue ${n}`, labels: ["bug"], createdAt: new Date(Date.now() - 3 * 86400_000).toISOString(),
  updatedAt: "2026-10-01T00:00:00Z", url: `https://github.com/o/r/issues/${n}`, ...over,
});

beforeEach(() => stubEventSource());
afterEach(() => vi.unstubAllGlobals());

describe("Issues", () => {
  it("lists issues with number, link, label and age", async () => {
    mockApi({ "GET /api/issues": { body: [issue(7, { labels: ["bug", "gpu"] })] } });
    render(<Issues />);
    const link = await screen.findByRole("link", { name: "Issue 7" });
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", "noopener noreferrer");
    expect(screen.getByText("#7")).toHaveClass("mono");
    expect(screen.getByText("bug")).toBeInTheDocument();
    expect(screen.getByText("gpu")).toBeInTheDocument();
    expect(screen.getByText("3 d ago")).toBeInTheDocument();
  });

  it("loading shows a skeleton; empty says how to file one; error shows the server message with Retry", async () => {
    mockApi({ "GET /api/issues": () => new Promise(() => {}) as never });
    const first = render(<Issues />);
    expect(screen.getByText("Loading Issues")).toBeInTheDocument();
    first.unmount();
    mockApi({ "GET /api/issues": { body: [] } });
    const second = render(<Issues />);
    expect(await screen.findByText(/No open issues\. Use New issue to file one\./)).toBeInTheDocument();
    second.unmount();
    mockApi({ "GET /api/issues": { status: 502, body: { error: "gh exited with code 1" } } });
    render(<Issues />);
    expect(await screen.findByRole("alert")).toHaveTextContent("gh exited with code 1");
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });

  it("validates the title on the client without calling the server, and moves focus to the field", async () => {
    const api = mockApi({ "GET /api/issues": { body: [] } });
    render(<Issues />);
    await userEvent.click(await screen.findByRole("button", { name: "New issue" }));
    const title = screen.getByLabelText("Title");
    expect(title).toHaveFocus();
    await userEvent.type(title, "   ");
    await userEvent.click(screen.getByRole("button", { name: "Create issue" }));
    expect(screen.getByText("Enter a title.")).toBeInTheDocument();
    expect(title).toHaveAttribute("aria-invalid", "true");
    expect(title).toHaveFocus();
    expect(api.calls.filter((c) => c.method === "POST")).toHaveLength(0);
  });

  it("rejects a title over the server limit", async () => {
    const api = mockApi({ "GET /api/issues": { body: [] } });
    render(<Issues />);
    await userEvent.click(await screen.findByRole("button", { name: "New issue" }));
    const title = screen.getByLabelText("Title");
    title.removeAttribute("maxlength");
    await userEvent.click(title);
    await userEvent.paste("x".repeat(201));
    await userEvent.click(screen.getByRole("button", { name: "Create issue" }));
    expect(screen.getByText(/at most 200 characters/)).toBeInTheDocument();
    expect(api.calls.filter((c) => c.method === "POST")).toHaveLength(0);
  });

  it("submits trimmed text, disables submit while pending, closes, refreshes and returns focus", async () => {
    const gate = deferred();
    const api = mockApi({
      "GET /api/issues": { body: [] },
      "POST /api/issues": async () => {
        await gate.promise;
        return { status: 201, body: { url: "https://github.com/o/r/issues/9" } };
      },
    });
    render(<Issues />);
    const open = await screen.findByRole("button", { name: "New issue" });
    await userEvent.click(open);
    await userEvent.type(screen.getByLabelText("Title"), "  Broken thing  ");
    await userEvent.type(screen.getByLabelText("Details (optional)"), "steps");
    await userEvent.click(screen.getByRole("button", { name: "Create issue" }));
    expect(screen.getByRole("button", { name: "Creating…" })).toBeDisabled();
    expect(api.calls.find((c) => c.method === "POST")?.body).toEqual({ title: "Broken thing", body: "steps" });
    gate.resolve();
    await waitFor(() => expect(screen.queryByLabelText("Title")).not.toBeInTheDocument());
    expect(screen.getByRole("status")).toHaveTextContent("Issue created");
    expect(screen.getByRole("button", { name: "New issue" })).toHaveFocus();
    await waitFor(() => expect(api.count("GET", "/api/issues")).toBe(2));
  });

  it("shows the server's 400 message next to the title field and keeps the form", async () => {
    mockApi({
      "GET /api/issues": { body: [] },
      "POST /api/issues": { status: 400, body: { error: "title must not contain NUL characters" } },
    });
    render(<Issues />);
    await userEvent.click(await screen.findByRole("button", { name: "New issue" }));
    await userEvent.type(screen.getByLabelText("Title"), "ok");
    await userEvent.click(screen.getByRole("button", { name: "Create issue" }));
    const title = screen.getByLabelText("Title");
    const msg = await screen.findByText("title must not contain NUL characters");
    expect(title).toHaveAccessibleDescription(/title must not contain NUL/);
    expect(msg).toBeInTheDocument();
    expect(title).toHaveValue("ok");
    expect(screen.getByRole("button", { name: "Create issue" })).toBeEnabled();
  });

  it("Cancel closes the form and returns focus to the New issue button", async () => {
    mockApi({ "GET /api/issues": { body: [] } });
    render(<Issues />);
    await userEvent.click(await screen.findByRole("button", { name: "New issue" }));
    await userEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(screen.queryByLabelText("Title")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "New issue" })).toHaveFocus();
  });
});
