import { describe, it, expect, afterEach, beforeEach, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Runs } from "../../web/src/panels/Runs";
import { mockApi, stubEventSource } from "./helpers";
import type { RunResult, RunRow, RunsResponse } from "../../shared/types";

beforeEach(() => stubEventSource());
afterEach(() => vi.unstubAllGlobals());

const result = (over: Partial<RunResult> = {}): RunResult => ({
  foldId: 1, testDetector: "agipd", inferenceThreshold: 0.5, cross: { ap: 0.8123, f1: 0.7 }, inDomain: { ap: 0.95 },
  track: null, probe: null, pretrainEpoch: null, inference: null, ...over,
});
const files = { bestPt: true, lastPt: false, resultsJson: true, resultsInferenceJson: false, wandbIdTxt: false, newestMtime: "2026-10-10T16:00:00.000Z" };
const run = (name: string, over: Partial<RunRow> = {}): RunRow => ({
  name, kind: "track1", generation: 2, archivedV1: false, status: "complete", files, result: result(), reevaluation: null, epochs: null, error: null, ...over,
});
const body = (runs: RunRow[], over: Partial<RunsResponse> = {}): RunsResponse => ({
  runs, scannedAt: "2026-10-10T16:00:00.000Z", checkpointsDirExists: true, truncated: false, error: null, ...over,
});
const provisional = run("vits16-mae-finetune-v2-fold2-seed42", {
  kind: "ssl-finetune", status: "provisional", epochs: { last: 187, configured: 200 },
  files: { ...files, resultsInferenceJson: true },
  result: result({ inference: { aggregation: "vote", checkpointEpoch: 187, evaluatedAt: "2026-10-10T16:00:00+00:00", trainingStatus: "incomplete", configuredEpochs: 200, lastLoggedEpoch: 187, wandbState: "crashed", warnings: ["training did not finish: wandb state crashed"] } }),
  reevaluation: result({ cross: { ap: 0.5 }, inference: { aggregation: "max", checkpointEpoch: 187, evaluatedAt: null, trainingStatus: "complete", configuredEpochs: 200, lastLoggedEpoch: 187, wandbState: "finished", warnings: [] } }),
});

describe("Runs", () => {
  it("shows one row per run with detector, metrics, epochs and an icon-plus-word status", async () => {
    mockApi({ "GET /api/runs": { body: body([run("resnet18-asymmetric-v2-fold1-seed42"), provisional]) } });
    render(<Runs />);
    const table = await screen.findByRole("table", { name: "Training runs" });
    const rows = within(table).getAllByRole("row");
    expect(rows).toHaveLength(3); // header + 2
    expect(within(rows[1]).getByText("resnet18-asymmetric-v2-fold1-seed42")).toHaveClass("mono");
    expect(within(rows[1]).getByText("Complete")).toBeInTheDocument();
    expect(within(rows[1]).getByText("agipd")).toBeInTheDocument();
    expect(within(rows[1]).getByText("0.812")).toBeInTheDocument();
    expect(within(rows[2]).getByText("Provisional")).toBeInTheDocument();
    expect(within(rows[2]).getByText("187 / 200")).toBeInTheDocument();
  });

  it("missing numbers read n/a, never NaN or blank", async () => {
    mockApi({ "GET /api/runs": { body: body([run("resnet18-asymmetric-v3-fold1-seed42", { result: result({ cross: null, inDomain: null, testDetector: null }) })]) } });
    render(<Runs />);
    await screen.findByRole("table");
    expect(screen.getAllByText("n/a").length).toBeGreaterThanOrEqual(3);
    expect(screen.queryByText(/NaN/)).toBeNull();
  });

  it("expanding a provisional fold shows its warnings verbatim; the re-evaluation is a second tab, not the result", async () => {
    mockApi({ "GET /api/runs": { body: body([provisional]) } });
    render(<Runs />);
    await userEvent.click(await screen.findByRole("button", { name: /vits16-mae-finetune-v2-fold2-seed42/ }));
    expect(screen.getByRole("tab", { name: "Result", selected: true })).toBeInTheDocument();
    expect(screen.getByText("training did not finish: wandb state crashed")).toBeInTheDocument();
    expect(within(screen.getByRole("tabpanel")).getByText("0.812")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("tab", { name: "Re-evaluation" }));
    expect(screen.getByRole("tabpanel")).toHaveTextContent("Re-evaluation, not the fold's result");
    expect(screen.getByRole("tabpanel")).toHaveTextContent("max");
    expect(within(screen.getByRole("tabpanel")).getByText("0.500")).toBeInTheDocument();
    expect(screen.queryByText("training did not finish: wandb state crashed")).toBeNull();
  });

  it("a run without a re-evaluation has no second tab", async () => {
    mockApi({ "GET /api/runs": { body: body([run("resnet18-asymmetric-v2-fold1-seed42")]) } });
    render(<Runs />);
    await userEvent.click(await screen.findByRole("button", { name: /resnet18-asymmetric-v2-fold1-seed42/ }));
    expect(screen.queryByRole("tab", { name: "Re-evaluation" })).toBeNull();
  });

  it("tabs work from the keyboard", async () => {
    mockApi({ "GET /api/runs": { body: body([provisional]) } });
    render(<Runs />);
    await userEvent.click(await screen.findByRole("button", { name: /vits16-mae-finetune/ }));
    screen.getByRole("tab", { name: "Result" }).focus();
    await userEvent.keyboard("{ArrowRight}");
    expect(screen.getByRole("tab", { name: "Re-evaluation", selected: true })).toHaveFocus();
  });

  it("legacy and archived runs are shown by default, labelled, and can be hidden", async () => {
    const legacy = run("resnet18-asymmetric-fold1-seed42", { kind: "legacy", generation: null });
    const archived = run("vits16-mae-finetune-fold2-seed42-v2", { kind: "legacy", generation: null, archivedV1: true });
    mockApi({ "GET /api/runs": { body: body([run("resnet18-asymmetric-v2-fold1-seed42"), legacy, archived]) } });
    render(<Runs />);
    await screen.findByRole("table");
    expect(screen.getAllByRole("row")).toHaveLength(4);
    expect(screen.getAllByText("Legacy name")).toHaveLength(2);
    expect(screen.getByText("Archived v1")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("checkbox", { name: "Show archived and legacy runs" }));
    expect(screen.getAllByRole("row")).toHaveLength(2);
  });

  it("filters by kind", async () => {
    mockApi({ "GET /api/runs": { body: body([run("resnet18-asymmetric-v2-fold1-seed42"), provisional]) } });
    render(<Runs />);
    await screen.findByRole("table");
    await userEvent.selectOptions(screen.getByLabelText("Kind"), "ssl-finetune");
    expect(screen.getAllByRole("row")).toHaveLength(2);
    expect(screen.queryByText("resnet18-asymmetric-v2-fold1-seed42")).toBeNull();
  });

  it("pretrain rows show the newest snapshot epoch and no metrics", async () => {
    mockApi({ "GET /api/runs": { body: body([run("mae-vits16-v2-fold1-seed42", { kind: "ssl-pretrain", status: "pretrain", result: null, epochs: { last: 400, configured: null } })]) } });
    render(<Runs />);
    await screen.findByRole("table");
    expect(screen.getByText("Pretrain checkpoints")).toBeInTheDocument();
    expect(screen.getByText("epoch 400")).toBeInTheDocument();
  });

  it("a row error is shown on the row and does not hide it", async () => {
    mockApi({ "GET /api/runs": { body: body([run("resnet50-asymmetric-v2-fold1-seed42", { status: "unknown", result: null, error: "results.json: not valid JSON" })]) } });
    render(<Runs />);
    await userEvent.click(await screen.findByRole("button", { name: /resnet50-asymmetric-v2-fold1-seed42/ }));
    expect(screen.getByRole("alert")).toHaveTextContent("results.json: not valid JSON");
  });

  it("run names from the filesystem are rendered as text, never as markup", async () => {
    const name = "<img src=x onerror=alert(1)>-asymmetric-v2-fold1-seed42";
    mockApi({ "GET /api/runs": { body: body([run(name, { kind: "legacy", generation: null })]) } });
    const { container } = render(<Runs />);
    await screen.findByRole("table");
    expect(container.querySelector("img")).toBeNull();
    expect(screen.getByText(name)).toBeInTheDocument();
  });

  it("loading, empty (no directory), empty (no runs) and error states", async () => {
    mockApi({ "GET /api/runs": () => new Promise(() => {}) as never });
    const a = render(<Runs />);
    expect(screen.getByText("Loading Runs")).toBeInTheDocument();
    a.unmount();
    mockApi({ "GET /api/runs": { body: body([], { checkpointsDirExists: false }) } });
    const b = render(<Runs />);
    expect(await screen.findByText(/No checkpoints directory on this machine/)).toBeInTheDocument();
    b.unmount();
    mockApi({ "GET /api/runs": { body: body([]) } });
    const c = render(<Runs />);
    expect(await screen.findByText(/No runs found under checkpoints\//)).toBeInTheDocument();
    c.unmount();
    mockApi({ "GET /api/runs": { status: 500, body: { error: "internal error" } } });
    render(<Runs />);
    expect(await screen.findByRole("alert")).toHaveTextContent("internal error");
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });

  it("says when the list was cut off at the scan limit", async () => {
    mockApi({ "GET /api/runs": { body: body([run("resnet18-asymmetric-v2-fold1-seed42")], { truncated: true }) } });
    render(<Runs />);
    expect(await screen.findByText(/Only the first 500 run directories are listed/)).toBeInTheDocument();
  });
});
