import { describe, it, expect, afterEach, beforeEach, vi } from "vitest";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { TestStatus } from "../../web/src/panels/TestStatus";
import { FakeEventSource, mockApi, stubEventSource } from "./helpers";
import type { CiRun, TestRunState } from "../../shared/types";

const run = (id: number, over: Partial<CiRun> = {}): CiRun => ({
  id, status: "completed", conclusion: "success", branch: "main", event: "push",
  createdAt: new Date(Date.now() - 2 * 3600_000).toISOString(), url: `https://github.com/o/r/actions/runs/${id}`, ...over,
});
const idle: TestRunState = { status: "idle", startedAt: null, finishedAt: null, commit: null, summary: null, tail: [], error: null };
const done: TestRunState = {
  ...idle, status: "done", startedAt: "2026-10-07T10:00:00Z", finishedAt: "2026-10-07T10:04:12Z", commit: "abc1234",
  summary: { tests: 20, passed: 17, failed: 2, errors: 0, skipped: 1, durationSec: 252, failures: [{ name: "tests/test_a.py::test_x", message: "assert 1 == 2" }, { name: "tests/test_b.py::test_y", message: "boom" }] },
  tail: [],
};
const running: TestRunState = { ...idle, status: "running", startedAt: "2026-10-07T10:00:00Z", tail: ["collecting", "tests/test_a.py ."] };
const ci = { body: { runs: [run(1), run(2, { conclusion: "failure" })] } };

beforeEach(() => stubEventSource());
afterEach(() => vi.unstubAllGlobals());

describe("TestStatus", () => {
  it("shows GitHub Actions and the local run side by side with counts, duration, commit and failures", async () => {
    mockApi({ "GET /api/ci": ci, "GET /api/tests": { body: done } });
    render(<TestStatus />);
    const gh = await screen.findByRole("region", { name: "GitHub Actions" });
    expect(within(gh).getByText("Passed")).toBeInTheDocument();
    expect(within(gh).getAllByRole("link", { name: "main" })[0]).toHaveAttribute("href", "https://github.com/o/r/actions/runs/1");
    expect(within(gh).getAllByText("2 h ago").length).toBeGreaterThan(0);
    const local = screen.getByRole("region", { name: "Local run" });
    expect(await within(local).findByText("17")).toHaveClass("mono");
    expect(within(local).getByText("2")).toHaveClass("mono");
    expect(within(local).getByText("4m 12s")).toBeInTheDocument();
    expect(within(local).getByText("abc1234")).toBeInTheDocument();
    expect(within(local).getByText("tests/test_a.py::test_x")).toBeInTheDocument();
    expect(within(local).getByText("assert 1 == 2")).toBeInTheDocument();
  });

  it("Run tests posts to the run endpoint then re-reads state", async () => {
    const api = mockApi({ "GET /api/ci": ci, "GET /api/tests": { body: idle }, "POST /api/tests/run": { status: 202, body: { started: true } } });
    render(<TestStatus />);
    await userEvent.click(await screen.findByRole("button", { name: "Run tests" }));
    await waitFor(() => expect(api.count("POST", "/api/tests/run")).toBe(1));
    await waitFor(() => expect(api.count("GET", "/api/tests")).toBe(2));
  });

  it("the button is disabled and reads Running while a run is active", async () => {
    mockApi({ "GET /api/ci": ci, "GET /api/tests": { body: running } });
    render(<TestStatus />);
    expect(await screen.findByRole("button", { name: "Running…" })).toBeDisabled();
  });

  it("409 says a run is already in progress", async () => {
    mockApi({ "GET /api/ci": ci, "GET /api/tests": { body: idle }, "POST /api/tests/run": { status: 409, body: { error: "A test run is already in progress" } } });
    render(<TestStatus />);
    await userEvent.click(await screen.findByRole("button", { name: "Run tests" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("A test run is already in progress");
  });

  it("streams tests-line events into the log and follows the tail unless the user scrolled up", async () => {
    mockApi({ "GET /api/ci": ci, "GET /api/tests": { body: running } });
    render(<TestStatus />);
    const log = await screen.findByLabelText("Test output");
    expect(log).toHaveTextContent("tests/test_a.py .");
    let top = 0;
    Object.defineProperty(log, "scrollHeight", { configurable: true, get: () => 1000 });
    Object.defineProperty(log, "clientHeight", { configurable: true, get: () => 100 });
    Object.defineProperty(log, "scrollTop", { configurable: true, get: () => top, set: (v: number) => { top = v; } });
    act(() => FakeEventSource.emit({ type: "tests-line", line: "tests/test_b.py F" }));
    expect(log).toHaveTextContent("tests/test_b.py F");
    expect(top).toBe(1000); // at the tail: follows
    top = 0;
    fireEvent.scroll(log); // user scrolled up
    act(() => FakeEventSource.emit({ type: "tests-line", line: "tests/test_c.py ." }));
    expect(log).toHaveTextContent("tests/test_c.py .");
    expect(top).toBe(0); // not yanked back
    await userEvent.click(screen.getByRole("button", { name: "Follow output" }));
    expect(top).toBe(1000);
  });

  it("announces the result politely when a run finishes", async () => {
    let state: TestRunState = running;
    mockApi({ "GET /api/ci": ci, "GET /api/tests": () => ({ body: state }) });
    render(<TestStatus />);
    await screen.findByRole("button", { name: "Running…" });
    const live = screen.getByRole("status");
    expect(live).toHaveAttribute("aria-live", "polite");
    expect(live).toHaveTextContent("");
    state = done;
    act(() => FakeEventSource.emit({ type: "invalidate", resource: "tests" }));
    await waitFor(() => expect(live).toHaveTextContent("Local tests finished: 17 passed, 2 failed"));
  });

  it("empty CI says so; a local error shows its text; each section has its own error and Retry", async () => {
    mockApi({
      "GET /api/ci": { body: { runs: [] } },
      "GET /api/tests": { body: { ...idle, status: "error", error: "pytest not found" } },
    });
    const first = render(<TestStatus />);
    expect(await screen.findByText("No CI runs found for this repository.")).toBeInTheDocument();
    expect(screen.getByText("pytest not found")).toBeInTheDocument();
    first.unmount();
    mockApi({ "GET /api/ci": { status: 502, body: { error: "gh failed" } }, "GET /api/tests": { body: idle } });
    render(<TestStatus />);
    const gh = await screen.findByRole("region", { name: "GitHub Actions" });
    expect(await within(gh).findByRole("alert")).toHaveTextContent("gh failed");
    expect(within(gh).getByRole("button", { name: "Retry GitHub Actions" })).toBeInTheDocument();
    expect(await screen.findByRole("button", { name: "Run tests" })).toBeInTheDocument();
  });

  it("loading shows skeletons in both sections", () => {
    mockApi({ "GET /api/ci": () => new Promise(() => {}) as never, "GET /api/tests": () => new Promise(() => {}) as never });
    render(<TestStatus />);
    expect(screen.getByText("Loading GitHub Actions")).toBeInTheDocument();
    expect(screen.getByText("Loading Local run")).toBeInTheDocument();
  });
});
