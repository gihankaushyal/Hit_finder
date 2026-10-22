import { describe, it, expect, afterEach, beforeEach, vi } from "vitest";
import { act, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { RunsTable as Runs } from "../../web/src/panels/RunsTable";
import { FakeEventSource, mockApi, stubEventSource } from "./helpers";
import type { RunResult, RunRow, RunsResponse } from "../../shared/types";

beforeEach(() => stubEventSource());
afterEach(() => vi.unstubAllGlobals());

const result = (over: Partial<RunResult> = {}): RunResult => ({
  foldId: 1, testDetector: "agipd", inferenceThreshold: 0.5, cross: { ap: 0.8123, f1: 0.7 }, inDomain: { ap: 0.95 },
  track: null, probe: null, pretrainEpoch: null, inference: null, training: null, ...over,
});
const files = { bestPt: true, lastPt: false, resultsJson: true, resultsInferenceJson: false, wandbIdTxt: false, newestMtime: "2026-10-10T16:00:00.000Z" };
const run = (name: string, over: Partial<RunRow> = {}): RunRow => ({
  name, kind: "track1", generation: 2, archivedV1: false, status: "complete", files, result: result(), reevaluation: null, epochs: null, error: null, ...over,
});
const body = (runs: RunRow[], over: Partial<RunsResponse> = {}): RunsResponse => ({
  runs, scannedAt: "2026-10-10T16:00:00.000Z", checkpointsDirExists: true, truncated: false, error: null, ...over,
});
const provisional = run("vits16-mae-finetune-v2-fold2-seed42", {
  kind: "ssl-finetune", status: "provisional", epochs: { last: 187, configured: 200, best: 187, source: "training-check" },
  files: { ...files, resultsInferenceJson: true },
  result: result({ inference: { aggregation: "vote", checkpointEpoch: 187, evaluatedAt: "2026-10-10T16:00:00+00:00", trainingStatus: "incomplete", configuredEpochs: 200, lastLoggedEpoch: 187, wandbState: "crashed", warnings: ["training did not finish: wandb state crashed"] } }),
  reevaluation: result({ cross: { ap: 0.5 }, inferenceThreshold: 0.25, inference: { aggregation: "max", checkpointEpoch: 187, evaluatedAt: null, trainingStatus: "complete", configuredEpochs: 200, lastLoggedEpoch: 187, wandbState: "finished", warnings: [] } }),
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

  it("long thresholds are rounded to three decimals in the detail view", async () => {
    mockApi({ "GET /api/runs": { body: body([run("resnet18-asymmetric-v2-fold1-seed42", { result: result({ inferenceThreshold: 0.02040816326530612 }) })]) } });
    render(<Runs />);
    await userEvent.click(await screen.findByRole("button", { name: /resnet18-asymmetric-v2-fold1-seed42/ }));
    const panel = screen.getByRole("tabpanel");
    expect(within(panel).getByText("0.020")).toBeInTheDocument();
    expect(panel).not.toHaveTextContent("0.0204081");
  });

  it("when a refresh drops the re-evaluation, the view falls back to the Result tab", async () => {
    let current = body([provisional]);
    mockApi({ "GET /api/runs": () => ({ body: current }) });
    render(<Runs />);
    await userEvent.click(await screen.findByRole("button", { name: /vits16-mae-finetune-v2-fold2-seed42/ }));
    await userEvent.click(screen.getByRole("tab", { name: "Re-evaluation" }));
    current = body([{ ...provisional, reevaluation: null, files: { ...files } }]);
    await act(async () => { FakeEventSource.emit({ type: "invalidate", resource: "runs" }); });
    expect(await screen.findByRole("tab", { name: "Result", selected: true })).toBeInTheDocument();
    expect(screen.getByText("training did not finish: wandb state crashed")).toBeInTheDocument();
    expect(screen.queryByText(/No results\.json for this run yet/)).toBeNull();
  });

  it("a directory read error is shown without also saying no runs were found", async () => {
    mockApi({ "GET /api/runs": { body: body([], { error: "the checkpoints directory could not be read" }) } });
    render(<Runs />);
    expect(await screen.findByRole("alert")).toHaveTextContent("the checkpoints directory could not be read");
    expect(screen.queryByText(/No runs found under checkpoints\//)).toBeNull();
  });

  it("when the filters hide every run, it says so instead of showing an empty table", async () => {
    mockApi({ "GET /api/runs": { body: body([run("resnet18-asymmetric-fold1-seed42", { kind: "legacy", generation: null })]) } });
    render(<Runs />);
    await screen.findByRole("table");
    await userEvent.click(screen.getByRole("checkbox", { name: "Show archived and legacy runs" }));
    expect(screen.getByText("No runs match the current filters.")).toBeInTheDocument();
    expect(screen.queryByRole("table")).toBeNull();
  });

  it("shows when each run was last updated, as a relative time with the exact time on hover", async () => {
    const recent = new Date(Date.now() - 3 * 3600_000).toISOString();
    mockApi({ "GET /api/runs": { body: body([run("resnet18-asymmetric-v2-fold1-seed42", { files: { ...files, newestMtime: recent } }), run("resnet18-asymmetric-v2-fold2-seed42", { files: { ...files, newestMtime: null } })]) } });
    render(<Runs />);
    const table = await screen.findByRole("table");
    expect(within(table).getByRole("columnheader", { name: "Updated" })).toBeInTheDocument();
    const rows = within(table).getAllByRole("row");
    expect(within(rows[1]).getByText("3 h ago")).toBeInTheDocument();
    expect(within(rows[1]).getByText("3 h ago").closest("time")).toHaveAttribute("dateTime", recent);
    expect(within(rows[2]).getAllByText("n/a").length).toBeGreaterThan(0);
  });

  describe("the Epochs cell", () => {
    const epochsRun = (epochs: RunRow["epochs"]) => run("resnet18-asymmetric-v2-fold1-seed42", { epochs });
    const renderOne = async (r: RunRow) => {
      mockApi({ "GET /api/runs": { body: body([r]) } });
      render(<Runs />);
      return within(await screen.findByRole("table"));
    };

    it("a recorded run shows last / configured, with the best epoch and an early stop on hover", async () => {
      const table = await renderOne(epochsRun({ last: 187, configured: 200, best: 150, source: "training" }));
      const cell = table.getByText("187 / 200");
      expect(cell).toHaveAttribute("title", expect.stringContaining("best epoch 150"));
    });

    it("a run whose trained epochs are unknown shows its best epoch instead of n/a, whatever the source", async () => {
      const table = await renderOne(epochsRun({ last: null, configured: 200, best: 12, source: "training" }));
      expect(table.getByText("best 12")).toBeInTheDocument();
    });

    it("an old run shows only the best epoch stored in best.pt, labelled as such", async () => {
      const table = await renderOne(epochsRun({ last: null, configured: null, best: 187, source: "checkpoint" }));
      const cell = table.getByText("best 187");
      expect(cell).toHaveAttribute("title", expect.stringContaining("not recorded"));
      expect(table.queryByText("n/a", { selector: "td:nth-child(7) *" })).toBeNull();
    });

    it("the detail view spells out epochs trained, whether it stopped early, and the best epoch", async () => {
      mockApi({ "GET /api/runs": { body: body([run("resnet18-asymmetric-v2-fold1-seed42", {
        epochs: { last: 187, configured: 200, best: 150, source: "training" },
        result: result({ training: { configuredEpochs: 200, lastEpoch: 187, bestEpoch: 150, earlyStopped: true } }),
      })]) } });
      render(<Runs />);
      await userEvent.click(await screen.findByRole("button", { name: /resnet18-asymmetric-v2-fold1-seed42/ }));
      const panel = within(screen.getByRole("tabpanel"));
      expect(panel.getByText("187 of 200, stopped early")).toBeInTheDocument();
      expect(panel.getByText("150")).toBeInTheDocument();
    });

    it("the detail view says a checkpoint-only epoch is the best epoch and that epochs trained were not recorded", async () => {
      mockApi({ "GET /api/runs": { body: body([epochsRun({ last: null, configured: null, best: 187, source: "checkpoint" })]) } });
      render(<Runs />);
      await userEvent.click(await screen.findByRole("button", { name: /resnet18-asymmetric-v2-fold1-seed42/ }));
      expect(within(screen.getByRole("tabpanel")).getByText(/187 \(from best\.pt; epochs trained were not recorded\)/)).toBeInTheDocument();
    });
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
    mockApi({ "GET /api/runs": { body: body([run("mae-vits16-v2-fold1-seed42", { kind: "ssl-pretrain", status: "pretrain", result: null, epochs: { last: 400, configured: null, best: null, source: "snapshot" } })]) } });
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
    expect(screen.getByText("Loading Active Runs")).toBeInTheDocument();
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

  describe("sorting", () => {
    const mk = (name: string, ap: number | null, over: Partial<RunRow> = {}) =>
      run(name, { result: result({ cross: ap === null ? null : { ap, f1: 1 - ap }, testDetector: name.toUpperCase() }), ...over });
    const runs3 = () => body([mk("b-run", 0.5), mk("a-run", 0.9), mk("c-run", null)]);
    const rowNames = () => screen.getAllByRole("row").slice(1).map((r) => within(r).getByRole("button", { name: /-run/ }).textContent);
    const header = (name: string) => screen.getByRole("columnheader", { name: new RegExp(`^${name}`) });

    it("opens ordered by name, with the Run column marked ascending", async () => {
      mockApi({ "GET /api/runs": { body: runs3() } });
      render(<Runs />);
      await screen.findByRole("table");
      expect(rowNames()).toEqual(["a-run", "b-run", "c-run"]);
      expect(header("Run")).toHaveAttribute("aria-sort", "ascending");
      expect(header("Cross AP")).toHaveAttribute("aria-sort", "none");
    });

    it("every requested column has a sort button", async () => {
      mockApi({ "GET /api/runs": { body: runs3() } });
      render(<Runs />);
      await screen.findByRole("table");
      for (const label of ["Run", "Status", "Held-out detector", "Cross AP", "Cross F1", "In-domain AP", "Epochs", "Updated"]) {
        expect(within(header(label)).getByRole("button", { name: new RegExp(`^${label}`) })).toBeInTheDocument();
      }
    });

    it("the first click on a metric sorts best first, the second reverses, and missing values stay last", async () => {
      mockApi({ "GET /api/runs": { body: runs3() } });
      render(<Runs />);
      await screen.findByRole("table");
      await userEvent.click(within(header("Cross AP")).getByRole("button"));
      expect(rowNames()).toEqual(["a-run", "b-run", "c-run"]);
      expect(header("Cross AP")).toHaveAttribute("aria-sort", "descending");
      expect(header("Run")).toHaveAttribute("aria-sort", "none");
      await userEvent.click(within(header("Cross AP")).getByRole("button"));
      expect(rowNames()).toEqual(["b-run", "a-run", "c-run"]);
      expect(header("Cross AP")).toHaveAttribute("aria-sort", "ascending");
    });

    it("sorting by status puts a provisional fold first", async () => {
      mockApi({ "GET /api/runs": { body: body([run("x-run"), provisional, run("y-run", { status: "no-results", result: null })]) } });
      render(<Runs />);
      await screen.findByRole("table");
      await userEvent.click(within(header("Status")).getByRole("button"));
      const first = screen.getAllByRole("row")[1];
      expect(within(first).getByText("Provisional")).toBeInTheDocument();
    });

    it("sorting works from the keyboard", async () => {
      mockApi({ "GET /api/runs": { body: runs3() } });
      render(<Runs />);
      await screen.findByRole("table");
      within(header("Cross F1")).getByRole("button").focus();
      await userEvent.keyboard("{Enter}");
      expect(header("Cross F1")).toHaveAttribute("aria-sort", "descending");
      await userEvent.keyboard(" ");
      expect(header("Cross F1")).toHaveAttribute("aria-sort", "ascending");
    });

    it("announces the new order to screen readers", async () => {
      mockApi({ "GET /api/runs": { body: runs3() } });
      render(<Runs />);
      await screen.findByRole("table");
      await userEvent.click(within(header("Cross AP")).getByRole("button"));
      expect(screen.getByRole("status")).toHaveTextContent("Sorted by Cross AP, highest first");
    });

    it("an expanded run stays open and attached to its row after the order changes", async () => {
      mockApi({ "GET /api/runs": { body: body([mk("b-run", 0.5, {}), mk("a-run", 0.9, {})]) } });
      render(<Runs />);
      await userEvent.click(await screen.findByRole("button", { name: "b-run" }));
      expect(screen.getByRole("tabpanel")).toBeInTheDocument();
      await userEvent.click(within(header("Cross AP")).getByRole("button"));
      const rows = screen.getAllByRole("row");
      expect(within(rows[1]).getByRole("button", { name: "a-run" })).toBeInTheDocument();
      expect(within(rows[2]).getByRole("button", { name: "b-run" })).toHaveAttribute("aria-expanded", "true");
      expect(rows[3]).toContainElement(screen.getByRole("tabpanel"));
    });

    it("keeps the chosen order when the list refreshes", async () => {
      mockApi({ "GET /api/runs": { body: runs3() } });
      render(<Runs />);
      await screen.findByRole("table");
      await userEvent.click(within(header("Cross AP")).getByRole("button"));
      await act(async () => { FakeEventSource.emit({ type: "invalidate", resource: "runs" }); });
      expect(header("Cross AP")).toHaveAttribute("aria-sort", "descending");
      expect(rowNames()).toEqual(["a-run", "b-run", "c-run"]);
    });
  });
});
