import fs from "node:fs";
import path from "node:path";

const metrics = (ap: number) => ({ ap, auc_roc: 0.9, f1: 0.7, threshold: 0.4 });
const base = (fold: number, detector: string, extra: object = {}) => ({
  fold_id: fold, test_detector: detector, inference_threshold: 0.5, cross: metrics(0.81), in_domain: metrics(0.95), ...extra,
});
const inference = (status: string | null, agg = "vote") => ({
  aggregation: agg, patch_stride: 224, min_hit_patches: 1, checkpoint_epoch: 187, evaluated_at: "2026-10-10T16:00:00+00:00",
  ...(status === null ? {} : {
    training_check: { status, checkpoint_epoch: 187, configured_epochs: 200, wandb_state: status === "incomplete" ? "crashed" : "finished",
      last_logged_epoch: 187, train_loss_flat: null, warnings: status === "incomplete" ? ["training did not finish: wandb state crashed"] : [] },
  }),
});

function put(root: string, run: string, files: Record<string, string | object>): void {
  const dir = path.join(root, run);
  fs.mkdirSync(dir, { recursive: true });
  for (const [name, content] of Object.entries(files)) {
    fs.writeFileSync(path.join(dir, name), typeof content === "string" ? content : JSON.stringify(content));
  }
}

/** Builds a temp `checkpoints/` covering every row type the Runs panel has to handle. Returns its path. */
export function buildCheckpoints(tmpRoot: string): string {
  const root = path.join(tmpRoot, "checkpoints");
  fs.mkdirSync(root, { recursive: true });
  put(root, "resnet18-asymmetric-v2-fold1-seed42", { "best.pt": "", "results.json": base(1, "agipd") });
  put(root, "resnet18-asymmetric-v2-fold2-seed42", { "best.pt": "" }); // no results yet
  put(root, "vits16-mae-finetune-v2-fold2-seed42", {
    "best.pt": "", "results.json": base(2, "epix10k", { track: "ssl", probe: false, pretrain_epoch: 400, inference: inference("incomplete") }),
    "results.inference.json": base(2, "epix10k", { inference: inference("complete", "max") }),
  });
  put(root, "vits16-mae-probe-v2-fold3-seed42", { "best.pt": "", "results.json": base(3, "jungfrau", { inference: inference("unknown") }) });
  put(root, "mae-vits16-v2-fold1-seed42", { "last.pt": "", "epoch20.pt": "", "epoch40.pt": "", "epoch100.pt": "" });
  put(root, "vits16-mae-finetune-fold2-seed42-v2", { PIPELINE_V1: "pipeline: v1\n", "best.pt": "", "results.json": base(2, "epix10k") }); // legacy name, v1 marker
  put(root, "resnet18-asymmetric-fold1-seed42", { "best.pt": "", "results.json": base(1, "agipd") }); // legacy, no marker
  put(root, "resnet50-asymmetric-v2-fold1-seed42", { "best.pt": "", "results.json": "{not json" }); // corrupt
  put(root, "resnet18-asymmetric-v3-fold1-seed42", { "best.pt": "", "results.json": { fold_id: 1 } }); // sparse
  put(root, "asymmetric_splits", { "split.txt": "a\n" }); // not a run
  fs.mkdirSync(path.join(root, "empty-dir"), { recursive: true }); // not a run
  const outside = path.join(tmpRoot, "outside");
  put(outside, ".", { "results.json": base(9, "x") });
  fs.symlinkSync(outside, path.join(root, "escape-link")); // symlink out of the tree
  return root;
}
