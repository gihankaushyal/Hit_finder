import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Skills } from "../../web/src/panels/Skills";
import { mockApi, stubEventSource } from "./helpers";

const NOW = Date.now();
const skills = [
  { name: "pipeline-check", description: "Smoke test", lastRun: new Date(NOW - 2 * 3600_000).toISOString(), confirmRequired: false, art: "pipeline" },
  { name: "slurm-job", description: "Submit a job", lastRun: null, confirmRequired: true, art: null },
];

let api: ReturnType<typeof mockApi>;
beforeEach(() => {
  stubEventSource();
  api = mockApi({ "GET /api/skills": { body: { skills } }, "POST /api/skills/pipeline-check/run": { status: 201, body: { name: "sk-pipeline-check-ab12" } },
    "POST /api/skills/slurm-job/run": { status: 201, body: { name: "sk-slurm-job-cd34" } } });
});
afterEach(() => vi.unstubAllGlobals());

describe("Skills", () => {
  it("shows a card per skill with when it last ran", async () => {
    render(<Skills />);
    expect(await screen.findByRole("button", { name: /pipeline-check/ })).toHaveTextContent("ran 2 h ago");
    expect(screen.getByRole("button", { name: /slurm-job/ })).toHaveTextContent("never ran");
  });

  it("expands on click and runs a read-only skill in one click", async () => {
    render(<Skills />);
    await userEvent.click(await screen.findByRole("button", { name: /pipeline-check/ }));
    expect(screen.getByText("Smoke test")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Run" }));
    const post = api.calls.find((c) => c.method === "POST");
    expect(post).toMatchObject({ path: "/api/skills/pipeline-check/run", body: { args: "", confirmed: false } });
    expect(await screen.findByRole("status")).toHaveTextContent("sk-pipeline-check-ab12");
  });

  it("asks for confirmation before a skill with side effects", async () => {
    render(<Skills />);
    await userEvent.click(await screen.findByRole("button", { name: /slurm-job/ }));
    await userEvent.type(screen.getByLabelText(/Arguments/), "configs/x.yaml");
    await userEvent.click(screen.getByRole("button", { name: "Run" }));
    expect(api.calls.some((c) => c.method === "POST")).toBe(false);
    await userEvent.click(screen.getByRole("button", { name: "Yes, run" }));
    const post = api.calls.find((c) => c.method === "POST");
    expect(post).toMatchObject({ path: "/api/skills/slurm-job/run", body: { args: "configs/x.yaml", confirmed: true } });
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
