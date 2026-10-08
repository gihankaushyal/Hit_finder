import { describe, it, expect, afterEach, beforeEach, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { LatestPr, FILES_COLLAPSED_COUNT } from "../../web/src/panels/LatestPr";
import { mockApi, stubEventSource } from "./helpers";
import type { PrDetail, PrSummary, PrsResponse } from "../../shared/types";

const summary = (n: number, over: Partial<PrSummary> = {}): PrSummary => ({
  number: n, title: `PR ${n}`, state: "MERGED", mergedAt: "2026-10-01T10:00:00Z",
  headRefName: `feat-${n}`, baseRefName: "main", additions: 10, deletions: 2, changedFiles: 3,
  url: `https://github.com/o/r/pull/${n}`, ...over,
});
const files = (n: number) => Array.from({ length: n }, (_, i) => ({ path: `src/file${i}.py`, additions: 1, deletions: 0 }));
const detail = (over: Partial<PrDetail> = {}): PrDetail => ({
  ...summary(42, { title: "Add frame cache", additions: 120, deletions: 15, changedFiles: 4 }),
  summary: ["Caches assembled frames", "Adds a build script"],
  testPlan: [{ text: "pytest passes", checked: true }, { text: "smoke on Sol", checked: false }],
  files: files(4), ...over,
});
const payload = (over: Partial<PrsResponse> = {}): PrsResponse => ({ latest: detail(), recent: [], open: [], ...over });

beforeEach(() => stubEventSource());
afterEach(() => vi.unstubAllGlobals());

describe("LatestPr", () => {
  it("shows a skeleton while loading", () => {
    mockApi({ "GET /api/prs": () => new Promise(() => {}) as never, "GET /api/ci": { body: { runs: [] } } });
    render(<LatestPr />);
    expect(screen.getByText("Loading Latest pull request")).toBeInTheDocument();
  });

  it("renders the latest PR: title link, number, head into base, stats, summary and test plan", async () => {
    mockApi({ "GET /api/prs": { body: payload() }, "GET /api/ci": { body: { runs: [] } } });
    render(<LatestPr />);
    const link = await screen.findByRole("link", { name: "Add frame cache" });
    expect(link).toHaveAttribute("href", "https://github.com/o/r/pull/42");
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", "noopener noreferrer");
    expect(screen.getByText("#42")).toHaveClass("mono");
    expect(screen.getByText("feat-42")).toHaveClass("mono");
    expect(screen.getByText("+120")).toBeInTheDocument();
    expect(screen.getByText("-15")).toBeInTheDocument();
    expect(screen.getByText((_, el) => el?.tagName === "SPAN" && el.textContent === "4 files")).toBeInTheDocument();
    const time = document.querySelector("time");
    expect(time).toHaveAttribute("datetime", "2026-10-01T10:00:00Z");
    expect(time).toHaveAttribute("title");
    const did = screen.getByRole("list", { name: "What it did" });
    expect(within(did).getAllByRole("listitem").map((l) => l.textContent)).toEqual(["Caches assembled frames", "Adds a build script"]);
    const plan = screen.getByRole("list", { name: "Test plan" });
    expect(within(plan).getByText("pytest passes").closest("s")).not.toBeNull();
    expect(within(plan).getByText("smoke on Sol").closest("s")).toBeNull();
    expect(within(plan).queryByRole("checkbox")).not.toBeInTheDocument();
    expect(screen.getByRole("heading", { level: 2, name: "Latest pull request" })).toBeInTheDocument();
  });

  it("renders PR text as text, never as HTML", async () => {
    const evil = '<img src=x onerror="alert(1)">';
    mockApi({
      "GET /api/prs": { body: payload({ latest: detail({ title: evil, summary: [evil] }) }) },
      "GET /api/ci": { body: { runs: [] } },
    });
    const { container } = render(<LatestPr />);
    await screen.findByRole("link", { name: evil });
    expect(container.querySelector("img")).toBeNull();
  });

  it("collapses a long file list behind a show all control", async () => {
    const total = FILES_COLLAPSED_COUNT + 5;
    mockApi({ "GET /api/prs": { body: payload({ latest: detail({ files: files(total), changedFiles: total }) }) }, "GET /api/ci": { body: { runs: [] } } });
    render(<LatestPr />);
    const list = await screen.findByRole("list", { name: "Touched files" });
    expect(within(list).getAllByRole("listitem")).toHaveLength(FILES_COLLAPSED_COUNT);
    const btn = screen.getByRole("button", { name: `Show all ${total} files` });
    expect(btn).toHaveAttribute("aria-expanded", "false");
    await userEvent.click(btn);
    expect(within(list).getAllByRole("listitem")).toHaveLength(total);
    expect(screen.getByRole("button", { name: "Show fewer files" })).toHaveAttribute("aria-expanded", "true");
  });

  it("lists open PRs first with their check state, and earlier PRs compactly", async () => {
    const api = mockApi({
      "GET /api/prs": {
        body: payload({
          open: [summary(50, { state: "OPEN", mergedAt: null, title: "WIP thing", headRefName: "wip", checks: { state: "failing", passing: 3, failing: 1, pending: 0 } })],
          recent: [summary(41, { title: "Older change" })],
        }),
      },
    });
    render(<LatestPr />);
    const open = await screen.findByRole("list", { name: "Open pull requests" });
    expect(within(open).getByRole("link", { name: "WIP thing" })).toBeInTheDocument();
    expect(within(open).getByText("Open")).toBeInTheDocument();
    expect(within(open).getByText("Checks failing (1 of 4)")).toBeInTheDocument();
    expect(api.count("GET", "/api/ci")).toBe(0); // the check state comes from /api/prs now
    const earlier = screen.getByRole("list", { name: "Earlier pull requests" });
    expect(within(earlier).getByRole("link", { name: "Older change" })).toBeInTheDocument();
    // open PRs come before the latest merged one
    expect(open.compareDocumentPosition(screen.getByRole("link", { name: "Add frame cache" })) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it.each([
    [{ state: "passing", passing: 3, failing: 0, pending: 0 }, "Checks passed (3)"],
    [{ state: "pending", passing: 1, failing: 0, pending: 2 }, "Checks running (2 of 3)"],
    [{ state: "none", passing: 0, failing: 0, pending: 0 }, "No checks reported"],
    [undefined, "No checks reported"],
  ] as const)("shows %j as %s", async (checks, text) => {
    mockApi({ "GET /api/prs": { body: payload({ open: [summary(50, { state: "OPEN", mergedAt: null, headRefName: "wip", ...(checks ? { checks } : {}) })] }) } });
    render(<LatestPr />);
    const open = await screen.findByRole("list", { name: "Open pull requests" });
    expect(within(open).getByText(text)).toBeInTheDocument();
  });

  it("empty: says what would appear", async () => {
    mockApi({ "GET /api/prs": { body: { latest: null, recent: [], open: [] } }, "GET /api/ci": { body: { runs: [] } } });
    render(<LatestPr />);
    expect(await screen.findByText(/No merged pull requests yet/)).toBeInTheDocument();
  });

  it("error: shows the server's error string and Retry fetches again", async () => {
    let n = 0;
    const api = mockApi({
      "GET /api/prs": () => (n++ === 0 ? { status: 502, body: { error: "GitHub CLI is not logged in" } } : { body: payload() }),
      "GET /api/ci": { body: { runs: [] } },
    });
    render(<LatestPr />);
    expect(await screen.findByRole("alert")).toHaveTextContent("GitHub CLI is not logged in");
    await userEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByRole("link", { name: "Add frame cache" })).toBeInTheDocument();
    expect(api.count("GET", "/api/prs")).toBe(2);
  });
});
