import { describe, it, expect } from "vitest";
import { classifyName, deriveStatus, parseResult } from "../server/lib/runParse";

describe("classifyName", () => {
  it.each([
    ["resnet18-asymmetric-v2-fold1-seed42", "track1", 2],
    ["mae-vits16-v2-fold3-seed42", "ssl-pretrain", 2],
    ["vits16-mae-finetune-v2-fold1-seed42", "ssl-finetune", 2],
    ["vits16-mae-probe-v3-fold4-seed7", "ssl-probe", 3],
    ["resnet18-asymmetric-v2-fold1-seed42-o2", "track1", 2],
  ])("%s is %s v%i", (name, kind, gen) => {
    expect(classifyName(name)).toEqual({ kind, generation: gen });
  });
  it.each([
    "resnet18-asymmetric-fold0-seed42",
    "resnet18-asymmetric-fold0-seed42-cache-smoketest-crops4",
    "mae-vits16-fold1-seed42",
    "vits16-mae-finetune-fold1-seed42-v2",
    "vits16-mae-probe-fold2-seed42-v2",
    "asymmetric_splits",
    "",
  ])("%j is legacy with no generation", (name) => {
    expect(classifyName(name)).toEqual({ kind: "legacy", generation: null });
  });
});

describe("parseResult", () => {
  const full = {
    fold_id: 2, test_detector: "agipd", inference_threshold: 0.5,
    cross: { ap: 0.8, auc_roc: 0.9, f1: 0.7, threshold: 0.4 }, in_domain: { ap: 0.95 },
    track: "ssl", probe: false, pretrain_checkpoint: "x", pretrain_epoch: 400,
    inference: {
      aggregation: "vote", patch_stride: 224, checkpoint_epoch: 187, evaluated_at: "2026-10-10T16:00:00+00:00",
      training_check: { status: "incomplete", configured_epochs: 200, last_logged_epoch: 187, wandb_state: "crashed", warnings: ["training did not finish"] },
    },
  };
  it("reads every documented key", () => {
    const r = parseResult(JSON.stringify(full));
    expect(r).toMatchObject({ foldId: 2, testDetector: "agipd", inferenceThreshold: 0.5, track: "ssl", probe: false, pretrainEpoch: 400 });
    expect(r.cross).toEqual({ ap: 0.8, auc_roc: 0.9, f1: 0.7, threshold: 0.4 });
    expect(r.inference).toEqual({
      aggregation: "vote", checkpointEpoch: 187, evaluatedAt: "2026-10-10T16:00:00+00:00", trainingStatus: "incomplete",
      configuredEpochs: 200, lastLoggedEpoch: 187, wandbState: "crashed", warnings: ["training did not finish"],
    });
  });
  it("treats everything except valid JSON objects as optional", () => {
    const r = parseResult('{"fold_id": 1}');
    expect(r).toMatchObject({ foldId: 1, testDetector: null, cross: null, inDomain: null, inference: null, track: null, probe: null });
  });
  it("turns non-numeric metric values into null and ignores nested values", () => {
    expect(parseResult('{"cross": {"ap": "x", "f1": 0.5, "nested": {"a": 1}}}').cross).toEqual({ ap: null, f1: 0.5, nested: null });
  });
  it("a __proto__ key in a metric block cannot change the block's prototype or become a metric", () => {
    const r = parseResult('{"cross": {"__proto__": {"x": 1}, "constructor": 2, "ap": 0.5}}');
    expect(Object.getPrototypeOf(r.cross)).toBe(Object.prototype);
    expect(Object.keys(r.cross!)).toEqual(["ap"]);
  });
  it("reads Python's NaN and Infinity tokens as missing numbers", () => {
    const r = parseResult('{"fold_id": 1, "cross": {"ap": NaN, "f1": 0.5, "x": -Infinity, "y": Infinity}, "in_domain": {"ap":NaN}}');
    expect(r.cross).toEqual({ ap: null, f1: 0.5, x: null, y: null });
    expect(r.inDomain).toEqual({ ap: null });
  });
  it("does not touch NaN inside strings, and still rejects other invalid JSON", () => {
    const r = parseResult('{"inference": {"training_check": {"status": "incomplete", "warnings": ["loss was NaN, stopping: NaN", "a \\" NaN"]}}}');
    expect(r.inference!.warnings).toEqual(["loss was NaN, stopping: NaN", 'a " NaN']);
    expect(() => parseResult('{"a": nan}')).toThrow("not valid JSON");
    expect(() => parseResult('{"a": NaNs}')).toThrow("not valid JSON");
  });
  it("reads the training block the training code writes", () => {
    const r = parseResult('{"training": {"configured_epochs": 200, "last_epoch": 187, "best_epoch": 150, "early_stopped": true}}');
    expect(r.training).toEqual({ configuredEpochs: 200, lastEpoch: 187, bestEpoch: 150, earlyStopped: true });
  });
  it("a missing training block is null, and a malformed one yields nulls rather than an error", () => {
    expect(parseResult("{}").training).toBeNull();
    expect(parseResult('{"training": 5}').training).toBeNull();
    expect(parseResult('{"training": {"last_epoch": "x", "best_epoch": null, "early_stopped": "yes"}}').training).toEqual({
      configuredEpochs: null, lastEpoch: null, bestEpoch: null, earlyStopped: null,
    });
  });
  it("an inference block without a training_check has no trainingStatus", () => {
    expect(parseResult('{"inference": {"aggregation": "max"}}').inference?.trainingStatus).toBeNull();
  });
  it("drops non-string warnings and caps their number and length", () => {
    const w = [1, "a".repeat(900), ...Array.from({ length: 40 }, (_, i) => `w${i}`)];
    const r = parseResult(JSON.stringify({ inference: { training_check: { status: "incomplete", warnings: w } } }));
    expect(r.inference!.warnings.length).toBe(20);
    expect(r.inference!.warnings[0].length).toBe(500);
  });
  it.each([["{not json", "not valid JSON"], ["[1,2]", "not a JSON object"], ["3", "not a JSON object"], ["null", "not a JSON object"]])(
    "%s throws a fixed message", (text, msg) => { expect(() => parseResult(text)).toThrow(msg); });
});

describe("deriveStatus", () => {
  const res = (status: string | null, withInference = true) =>
    parseResult(JSON.stringify(withInference ? { inference: status === null ? {} : { training_check: { status } } } : {}));
  it("pretrain directories are pretrain whatever their files", () => {
    expect(deriveStatus("ssl-pretrain", { resultsJson: false, anyCheckpoint: true, pretrainShaped: false }, null, false)).toBe("pretrain");
  });
  it("a legacy-named directory shaped like a pretrain run (no best.pt, no results) is pretrain", () => {
    expect(deriveStatus("legacy", { resultsJson: false, anyCheckpoint: true, pretrainShaped: true }, null, false)).toBe("pretrain");
  });
  it("a checkpoint with no results.json has no results", () => {
    expect(deriveStatus("track1", { resultsJson: false, anyCheckpoint: true, pretrainShaped: false }, null, false)).toBe("no-results");
  });
  it("a results.json without an inference block is complete", () => {
    expect(deriveStatus("track1", { resultsJson: true, anyCheckpoint: true, pretrainShaped: false }, res(null, false), false)).toBe("complete");
  });
  it.each([["complete", "complete"], ["incomplete", "provisional"], ["unknown", "unknown"]])("training_check %s gives %s", (tc, expected) => {
    expect(deriveStatus("track1", { resultsJson: true, anyCheckpoint: true, pretrainShaped: false }, res(tc), false)).toBe(expected);
  });
  it("an inference block with no training_check is unknown", () => {
    expect(deriveStatus("track1", { resultsJson: true, anyCheckpoint: true, pretrainShaped: false }, res(null), false)).toBe("unknown");
  });
  it("a results.json that could not be parsed is unknown", () => {
    expect(deriveStatus("track1", { resultsJson: true, anyCheckpoint: true, pretrainShaped: false }, null, true)).toBe("unknown");
  });
});
