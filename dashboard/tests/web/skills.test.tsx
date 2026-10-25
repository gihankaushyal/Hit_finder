import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Skills } from "../../web/src/panels/Skills";
import { mockApi, stubEventSource } from "./helpers";

const NOW = Date.now();
const skills = [
  { name: "pipeline-check", description: "Smoke test", lastRun: new Date(NOW - 2 * 3600_000).toISOString(), art: "pipeline" },
  { name: "slurm-job", description: "Submit a job", lastRun: null, art: null },
];

let api: ReturnType<typeof mockApi>;
beforeEach(() => {
  stubEventSource();
  api = mockApi({ "GET /api/skills": { body: { skills } } });
});
afterEach(() => vi.unstubAllGlobals());

describe("Skills", () => {
  it("shows a card per skill with when it last ran", async () => {
    render(<Skills />);
    expect(await screen.findByRole("button", { name: /pipeline-check/ })).toHaveTextContent("ran 2 h ago");
    expect(screen.getByRole("button", { name: /slurm-job/ })).toHaveTextContent("never ran");
  });

  it("expands on click to the description and how to run it from Claude Code, with no Run button", async () => {
    render(<Skills />);
    await userEvent.click(await screen.findByRole("button", { name: /pipeline-check/ }));
    expect(screen.getByText("Smoke test")).toBeInTheDocument();
    expect(screen.getByText("/pipeline-check")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^Run/ })).toBeNull();
    expect(api.calls.some((c) => c.method === "POST")).toBe(false);
  });

  it("keeps the cards in a real list and shows one detail panel below it", async () => {
    render(<Skills />);
    await userEvent.click(await screen.findByRole("button", { name: /pipeline-check/ }));
    expect(screen.getAllByRole("listitem")).toHaveLength(2);
    expect(screen.getByRole("button", { name: /pipeline-check/ })).toHaveAttribute("aria-controls", "skill-detail");
    expect(screen.getByRole("button", { name: /slurm-job/ })).not.toHaveAttribute("aria-controls");
    expect(document.querySelectorAll("#skill-detail")).toHaveLength(1);
    await userEvent.click(screen.getByRole("button", { name: /pipeline-check/ }));
    expect(document.getElementById("skill-detail")).toBeNull();
  });
});
