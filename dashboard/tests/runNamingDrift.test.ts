import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { classifyName } from "../server/lib/runParse";

// The dashboard never imports src/. It copies the run-name patterns, so this test reads the Python
// file as plain text and checks that its own examples still classify. It runs where src/ exists
// (the phase-05 branch and later); RUN_NAMING_PY points it at another copy.
const PY = process.env.RUN_NAMING_PY ?? path.resolve(__dirname, "..", "..", "src", "training", "run_naming.py");
const text = fs.existsSync(PY) ? fs.readFileSync(PY, "utf8") : null;
const example = (name: string): string => {
  const m = new RegExp(`^${name} = "([^"]+)"`, "m").exec(text!);
  if (!m) throw new Error(`${name} not found in run_naming.py`);
  return m[1];
};

describe.skipIf(text === null)("run name patterns still agree with src/training/run_naming.py", () => {
  it("the Track 1 example is a track1 run", () => {
    expect(classifyName(`${example("ASYMMETRIC_EXAMPLE")}-fold1-seed42`).kind).toBe("track1");
  });
  it("the SSL pretrain example is an ssl-pretrain run", () => {
    expect(classifyName(`${example("SSL_PRETRAIN_EXAMPLE")}-fold1-seed42`).kind).toBe("ssl-pretrain");
  });
  it("the fine-tune example, expanded the way expand_finetune_prefix does, is a finetune and a probe run", () => {
    const m = /^([a-z0-9]+)-mae-v([0-9]+)$/.exec(example("SSL_FINETUNE_EXAMPLE"))!;
    expect(classifyName(`${m[1]}-mae-finetune-v${m[2]}-fold1-seed42`)).toEqual({ kind: "ssl-finetune", generation: Number(m[2]) });
    expect(classifyName(`${m[1]}-mae-probe-v${m[2]}-fold1-seed42`)).toEqual({ kind: "ssl-probe", generation: Number(m[2]) });
  });
  it("the file still builds run names as <prefix>-fold<N>-seed<S><suffix>", () => {
    expect(text).toContain('f"{prefix}-fold{fold_id}-seed{seed}{run_suffix}"');
    expect(text).toContain('f"{m[\'backbone\']}-mae-{mode}-v{m[\'version\']}"');
  });
});
