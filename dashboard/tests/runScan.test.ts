import fs from "node:fs";
import fsp from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createRunScanner, MAX_RUN_DIRS, scanRuns } from "../server/lib/runScan";
import type { RunsResponse } from "../shared/types";
import { buildCheckpoints } from "./helpers/runFixtures";

let tmp: string;
beforeEach(() => { tmp = fs.mkdtempSync(path.join(os.tmpdir(), "runscan-")); });
afterEach(() => { vi.restoreAllMocks(); fs.rmSync(tmp, { recursive: true, force: true }); });

const row = (res: RunsResponse, name: string) => res.runs.find((r) => r.name === name)!;

describe("scanRuns", () => {
  it("lists run directories only: splits, empty dirs and symlinks are skipped", async () => {
    const res = await scanRuns(buildCheckpoints(tmp));
    const names = res.runs.map((r) => r.name);
    expect(names).not.toContain("asymmetric_splits");
    expect(names).not.toContain("empty-dir");
    expect(names).not.toContain("escape-link");
    expect(names).toHaveLength(9);
    expect(names).toEqual([...names].sort((a, b) => a.localeCompare(b, "en", { numeric: true })));
    expect(res.checkpointsDirExists).toBe(true);
    expect(res.truncated).toBe(false);
  });

  it("a complete Track 1 run carries its held-out detector and metrics", async () => {
    const r = row(await scanRuns(buildCheckpoints(tmp)), "resnet18-asymmetric-v2-fold1-seed42");
    expect(r).toMatchObject({ kind: "track1", generation: 2, archivedV1: false, status: "complete", error: null, reevaluation: null });
    expect(r.result).toMatchObject({ foldId: 1, testDetector: "agipd" });
    expect(r.result!.cross).toMatchObject({ ap: 0.81 });
    expect(r.files).toMatchObject({ bestPt: true, resultsJson: true, resultsInferenceJson: false });
    expect(r.epochs).toEqual({ last: null, configured: null, best: 187, source: "checkpoint" });
    expect(r.files.newestMtime).toMatch(/^\d{4}-\d\d-\d\dT/);
  });

  it("a provisional fold is flagged and its re-evaluation is kept separate", async () => {
    const r = row(await scanRuns(buildCheckpoints(tmp)), "vits16-mae-finetune-v2-fold2-seed42");
    expect(r.kind).toBe("ssl-finetune");
    expect(r.status).toBe("provisional");
    expect(r.result!.inference!.warnings).toEqual(["training did not finish: wandb state crashed"]);
    expect(r.epochs).toEqual({ last: 187, configured: 200, best: 187, source: "training-check" });
    expect(r.reevaluation!.inference!.aggregation).toBe("max");
    expect(r.result!.inference!.aggregation).toBe("vote");
  });

  it("unknown training check is neutral, not an error", async () => {
    const r = row(await scanRuns(buildCheckpoints(tmp)), "vits16-mae-probe-v2-fold3-seed42");
    expect(r).toMatchObject({ kind: "ssl-probe", status: "unknown", error: null });
  });

  it("a checkpoint with no results.json has no results", async () => {
    expect(row(await scanRuns(buildCheckpoints(tmp)), "resnet18-asymmetric-v2-fold2-seed42")).toMatchObject({ status: "no-results", result: null });
  });

  it("pretrain rows report the newest epoch snapshot, numerically", async () => {
    const r = row(await scanRuns(buildCheckpoints(tmp)), "mae-vits16-v2-fold1-seed42");
    expect(r).toMatchObject({ kind: "ssl-pretrain", status: "pretrain" });
    expect(r.epochs).toEqual({ last: 100, configured: null, best: null, source: "snapshot" });
  });

  it("the PIPELINE_V1 marker wins over a -v2 name; the legacy name is kept", async () => {
    const r = row(await scanRuns(buildCheckpoints(tmp)), "vits16-mae-finetune-fold2-seed42-v2");
    expect(r).toMatchObject({ kind: "legacy", generation: null, archivedV1: true, status: "complete" });
  });

  it("a corrupt results.json gives a row with a fixed error, never a throw", async () => {
    const r = row(await scanRuns(buildCheckpoints(tmp)), "resnet50-asymmetric-v2-fold1-seed42");
    expect(r).toMatchObject({ status: "unknown", result: null });
    expect(r.error).toBe("results.json: not valid JSON");
    expect(JSON.stringify(r)).not.toContain(tmp);
  });

  it("a sparse results.json gives nulls, not errors", async () => {
    const r = row(await scanRuns(buildCheckpoints(tmp)), "resnet18-asymmetric-v3-fold1-seed42");
    expect(r).toMatchObject({ status: "complete", error: null });
    expect(r.result).toMatchObject({ foldId: 1, testDetector: null, cross: null });
  });

  it("a symlinked results.json inside a run is not followed", async () => {
    const root = buildCheckpoints(tmp);
    const dir = path.join(root, "resnet18-asymmetric-v2-fold1-seed42");
    fs.rmSync(path.join(dir, "results.json"));
    fs.symlinkSync(path.join(tmp, "outside", "results.json"), path.join(dir, "results.json"));
    const r = row(await scanRuns(root), "resnet18-asymmetric-v2-fold1-seed42");
    expect(r.files.resultsJson).toBe(false);
    expect(r.result).toBeNull();
  });

  it("an oversized results.json is refused with a fixed message", async () => {
    const root = buildCheckpoints(tmp);
    fs.writeFileSync(path.join(root, "resnet18-asymmetric-v2-fold1-seed42", "results.json"), "x".repeat(1_000_001));
    expect(row(await scanRuns(root), "resnet18-asymmetric-v2-fold1-seed42").error).toBe("results.json: larger than the size limit");
  });

  it("a results.json written by Python with NaN metrics parses, with those metrics missing", async () => {
    const root = buildCheckpoints(tmp);
    fs.writeFileSync(path.join(root, "resnet18-asymmetric-v3-fold1-seed42", "results.json"), '{"fold_id": 1, "cross": {"ap": NaN, "f1": 0.5}}');
    const r = row(await scanRuns(root), "resnet18-asymmetric-v3-fold1-seed42");
    expect(r).toMatchObject({ status: "complete", error: null });
    expect(r.result!.cross).toEqual({ ap: null, f1: 0.5 });
  });

  it("an old-style pretrain directory (legacy name, last.pt and epoch snapshots only) is a pretrain row with its newest epoch", async () => {
    const root = buildCheckpoints(tmp);
    const dir = path.join(root, "mae-vits16-fold1-seed42");
    fs.mkdirSync(dir);
    for (const f of ["last.pt", "epoch20.pt", "epoch400.pt", "PIPELINE_V1"]) fs.writeFileSync(path.join(dir, f), "");
    const r = row(await scanRuns(root), "mae-vits16-fold1-seed42");
    expect(r).toMatchObject({ kind: "legacy", archivedV1: true, status: "pretrain", epochs: { last: 400, configured: null, best: null, source: "snapshot" } });
  });

  it("a legacy directory with only best.pt still has no results", async () => {
    const root = buildCheckpoints(tmp);
    const dir = path.join(root, "vits16-mae-probe-fold4-seed42");
    fs.mkdirSync(dir);
    fs.writeFileSync(path.join(dir, "best.pt"), "");
    expect(row(await scanRuns(root), "vits16-mae-probe-fold4-seed42").status).toBe("no-results");
  });

  it("an in-flight fine-tune, probe or Track 1 run with last.pt but no best.pt is not a pretrain row", async () => {
    const root = buildCheckpoints(tmp);
    for (const name of ["vits16-mae-finetune-v2-fold1-seed42", "vits16-mae-probe-v2-fold1-seed42", "resnet18-asymmetric-v4-fold1-seed42"]) {
      fs.mkdirSync(path.join(root, name));
      fs.writeFileSync(path.join(root, name, "last.pt"), "");
    }
    const res = await scanRuns(root);
    for (const name of ["vits16-mae-finetune-v2-fold1-seed42", "vits16-mae-probe-v2-fold1-seed42", "resnet18-asymmetric-v4-fold1-seed42"]) {
      expect(row(res, name)).toMatchObject({ status: "no-results", epochs: null });
    }
  });

  it("orders generations numerically, so v2 comes before v10", async () => {
    const root = buildCheckpoints(tmp);
    for (const g of [10, 2]) {
      const dir = path.join(root, `resnet18-asymmetric-v${g}-fold7-seed42`);
      fs.mkdirSync(dir);
      fs.writeFileSync(path.join(dir, "best.pt"), "");
    }
    const names = (await scanRuns(root)).runs.map((r) => r.name);
    expect(names.indexOf("resnet18-asymmetric-v2-fold7-seed42")).toBeLessThan(names.indexOf("resnet18-asymmetric-v10-fold7-seed42"));
  });

  it("newestMtime counts the newest epoch snapshot of a pretrain directory", async () => {
    const root = buildCheckpoints(tmp);
    const dir = path.join(root, "mae-vits16-v2-fold1-seed42");
    const old = new Date("2026-01-01T00:00:00Z");
    const recent = new Date("2026-10-01T12:00:00Z");
    for (const f of ["last.pt", "epoch20.pt", "epoch40.pt", "epoch100.pt"]) fs.utimesSync(path.join(dir, f), old, old);
    fs.utimesSync(path.join(dir, "epoch100.pt"), recent, recent);
    expect((await scanRuns(root)).runs.find((r) => r.name === "mae-vits16-v2-fold1-seed42")!.files.newestMtime).toBe(recent.toISOString());
  });

  describe("the Epochs value comes from the best available source", () => {
    const FIXTURES = path.join(__dirname, "fixtures", "ckpt");
    const makeRun = (root: string, name: string, files: Record<string, string | Buffer>) => {
      const dir = path.join(root, name);
      fs.mkdirSync(dir, { recursive: true });
      for (const [f, content] of Object.entries(files)) fs.writeFileSync(path.join(dir, f), content);
      return dir;
    };
    const bestFixture = (n: number) => fs.readFileSync(path.join(FIXTURES, `best-${n}.pt`));
    const training = { configured_epochs: 200, last_epoch: 187, best_epoch: 150, early_stopped: true };

    it("the training block in results.json wins, and best.pt is not even opened", async () => {
      const root = path.join(tmp, "checkpoints");
      makeRun(root, "resnet18-asymmetric-v2-fold1-seed42", { "best.pt": bestFixture(5), "results.json": JSON.stringify({ fold_id: 1, training }) });
      const open = vi.spyOn(fsp, "open");
      const r = row(await scanRuns(root), "resnet18-asymmetric-v2-fold1-seed42");
      expect(r.epochs).toEqual({ last: 187, configured: 200, best: 150, source: "training" });
      expect(r.result!.training).toMatchObject({ earlyStopped: true });
      expect(open.mock.calls.filter((c) => String(c[0]).endsWith("best.pt"))).toHaveLength(0);
    });

    it("without it, a run that has only best.pt shows the epoch stored in the checkpoint", async () => {
      const root = path.join(tmp, "checkpoints");
      makeRun(root, "resnet18-asymmetric-v2-fold2-seed42", { "best.pt": bestFixture(187), "results.json": JSON.stringify({ fold_id: 2 }) });
      const r = row(await scanRuns(root), "resnet18-asymmetric-v2-fold2-seed42");
      expect(r.status).toBe("complete");
      expect(r.epochs).toEqual({ last: null, configured: null, best: 187, source: "checkpoint" });
    });

    it("a run with a checkpoint but no results yet still shows its best epoch", async () => {
      const root = path.join(tmp, "checkpoints");
      makeRun(root, "resnet18-asymmetric-v2-fold3-seed42", { "best.pt": bestFixture(5) });
      const r = row(await scanRuns(root), "resnet18-asymmetric-v2-fold3-seed42");
      expect(r).toMatchObject({ status: "no-results", epochs: { last: null, configured: null, best: 5, source: "checkpoint" } });
    });

    it("an unreadable best.pt means no epochs and no error on the row", async () => {
      const root = path.join(tmp, "checkpoints");
      makeRun(root, "resnet18-asymmetric-v2-fold4-seed42", { "best.pt": Buffer.alloc(3000, 9), "results.json": JSON.stringify({ fold_id: 4 }) });
      const r = row(await scanRuns(root), "resnet18-asymmetric-v2-fold4-seed42");
      expect(r.epochs).toBeNull();
      expect(r.error).toBeNull();
    });

    it("a rescan does not read an unchanged best.pt again", async () => {
      const root = path.join(tmp, "checkpoints");
      makeRun(root, "resnet18-asymmetric-v2-fold5-seed42", { "best.pt": bestFixture(187) });
      await scanRuns(root);
      const open = vi.spyOn(fsp, "open");
      const r = row(await scanRuns(root), "resnet18-asymmetric-v2-fold5-seed42");
      expect(r.epochs?.best).toBe(187);
      expect(open.mock.calls.filter((c) => String(c[0]).endsWith("best.pt"))).toHaveLength(0);
    });
  });

  it("a missing checkpoints directory is an empty result, not an error", async () => {
    expect(await scanRuns(path.join(tmp, "nope"))).toMatchObject({ runs: [], checkpointsDirExists: false, error: null });
  });

  it("caps the number of listed directories", async () => {
    const root = path.join(tmp, "checkpoints");
    for (let i = 0; i < MAX_RUN_DIRS + 3; i++) {
      fs.mkdirSync(path.join(root, `run-${String(i).padStart(4, "0")}`), { recursive: true });
      fs.writeFileSync(path.join(root, `run-${String(i).padStart(4, "0")}`, "best.pt"), "");
    }
    const res = await scanRuns(root);
    expect(res.runs).toHaveLength(MAX_RUN_DIRS);
    expect(res.truncated).toBe(true);
  });
});

describe("createRunScanner", () => {
  it("serves a cached scan inside the TTL and rescans after it", async () => {
    const root = buildCheckpoints(tmp);
    let t = 1_000;
    const s = createRunScanner(root, { ttlMs: 10_000, now: () => t });
    const first = await s.scan();
    fs.mkdirSync(path.join(root, "resnet18-asymmetric-v9-fold1-seed42"));
    fs.writeFileSync(path.join(root, "resnet18-asymmetric-v9-fold1-seed42", "best.pt"), "");
    t += 9_999;
    expect(await s.scan()).toBe(first);
    t += 2;
    expect((await s.scan()).runs.length).toBe(first.runs.length + 1);
  });

  const empty = (): RunsResponse => ({ runs: [], scannedAt: "", checkpointsDirExists: true, truncated: false, error: null });

  it("concurrent requests share one scan instead of starting one each", async () => {
    let calls = 0;
    let release!: () => void;
    const gate = new Promise<void>((r) => { release = r; });
    const s = createRunScanner("/unused", { scanFn: async () => { calls++; await gate; return empty(); } });
    const all = Promise.all([s.scan(), s.scan(), s.scan()]);
    release();
    const [a, b, c] = await all;
    expect(calls).toBe(1);
    expect(a).toBe(b);
    expect(b).toBe(c);
  });

  it("a failed scan is not cached: the next request tries again", async () => {
    let calls = 0;
    const s = createRunScanner("/unused", { scanFn: async () => { calls++; if (calls === 1) throw new Error("boom"); return empty(); } });
    await expect(s.scan()).rejects.toThrow("boom");
    await expect(s.scan()).resolves.toMatchObject({ checkpointsDirExists: true });
    expect(calls).toBe(2);
  });

  it("does not block the event loop while it scans", async () => {
    // Asynchronous fs calls hand control back to the event loop; a synchronous scan would finish with zero turns.
    const root = buildCheckpoints(tmp);
    let turns = 0;
    let done = false;
    const spin = () => { if (!done) { turns++; setImmediate(spin); } };
    setImmediate(spin);
    await scanRuns(root);
    done = true;
    expect(turns).toBeGreaterThan(0);
  });

});
