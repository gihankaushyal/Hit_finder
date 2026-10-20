import { describe, expect, it } from "vitest";
import { SORT_KEYS, defaultDirection, sortRuns, type SortKey } from "../../web/src/lib/runSort";
import type { RunResult, RunRow, RunStatus } from "../../shared/types";

const result = (over: Partial<RunResult> = {}): RunResult => ({
  foldId: 1, testDetector: "AGIPD", inferenceThreshold: 0.5, cross: { ap: 0.5, f1: 0.5 }, inDomain: { ap: 0.5 },
  track: null, probe: null, pretrainEpoch: null, inference: null, training: null, ...over,
});
const row = (name: string, over: Partial<RunRow> = {}): RunRow => ({
  name, kind: "track1", generation: 2, archivedV1: false, status: "complete",
  files: { bestPt: true, lastPt: false, resultsJson: true, resultsInferenceJson: false, wandbIdTxt: false, newestMtime: null },
  result: result(), reevaluation: null, epochs: null, error: null, ...over,
});
const names = (rs: RunRow[]) => rs.map((r) => r.name);
const withCross = (name: string, ap: number | null) => row(name, { result: result({ cross: ap === null ? null : { ap, f1: ap } }) });

describe("sortRuns", () => {
  it("numbers: descending puts the largest first, ascending the smallest", () => {
    const rs = [withCross("a", 0.5), withCross("b", 0.9), withCross("c", 0.1)];
    expect(names(sortRuns(rs, "crossAp", "desc"))).toEqual(["b", "a", "c"]);
    expect(names(sortRuns(rs, "crossAp", "asc"))).toEqual(["c", "a", "b"]);
  });

  it("missing values come last in both directions", () => {
    const rs = [withCross("none", null), withCross("lo", 0.1), withCross("hi", 0.9)];
    expect(names(sortRuns(rs, "crossAp", "desc"))).toEqual(["hi", "lo", "none"]);
    expect(names(sortRuns(rs, "crossAp", "asc"))).toEqual(["lo", "hi", "none"]);
  });

  it("cross F1 and in-domain AP read their own metric", () => {
    const a = row("a", { result: result({ cross: { ap: 0.1, f1: 0.9 }, inDomain: { ap: 0.2 } }) });
    const b = row("b", { result: result({ cross: { ap: 0.9, f1: 0.1 }, inDomain: { ap: 0.8 } }) });
    expect(names(sortRuns([a, b], "crossF1", "desc"))).toEqual(["a", "b"]);
    expect(names(sortRuns([a, b], "crossAp", "desc"))).toEqual(["b", "a"]);
    expect(names(sortRuns([a, b], "inDomainAp", "desc"))).toEqual(["b", "a"]);
  });

  it("status ascending puts runs that need a human first: provisional, unknown, no results, pretrain, complete", () => {
    const order: RunStatus[] = ["complete", "pretrain", "no-results", "unknown", "provisional"];
    const rs = order.map((s) => row(s, { status: s }));
    expect(names(sortRuns(rs, "status", "asc"))).toEqual(["provisional", "unknown", "no-results", "pretrain", "complete"]);
    expect(names(sortRuns(rs, "status", "desc"))).toEqual(["complete", "pretrain", "no-results", "unknown", "provisional"]);
  });

  it("held-out detector sorts as text, ignoring case, with missing last", () => {
    const rs = [row("n", { result: null }), row("e", { result: result({ testDetector: "ePix10k" }) }), row("a", { result: result({ testDetector: "AGIPD" }) }), row("j", { result: result({ testDetector: "JUNGFRAU_4M" }) })];
    expect(names(sortRuns(rs, "detector", "asc"))).toEqual(["a", "e", "j", "n"]);
    expect(names(sortRuns(rs, "detector", "desc"))).toEqual(["j", "e", "a", "n"]);
  });

  it("epochs sort by the last epoch", () => {
    const rs = [row("x", { epochs: { last: 10, configured: 200, best: 10, source: "training" } }), row("y", { epochs: { last: 187, configured: 200, best: 187, source: "training" } }), row("z", { epochs: null })];
    expect(names(sortRuns(rs, "epochs", "desc"))).toEqual(["y", "x", "z"]);
    expect(names(sortRuns(rs, "epochs", "asc"))).toEqual(["x", "y", "z"]);
  });

  it("epochs fall back to the best epoch when the number trained is not recorded", () => {
    const rs = [
      row("trained", { epochs: { last: 100, configured: 200, best: 90, source: "training" } }),
      row("old", { epochs: { last: null, configured: null, best: 150, source: "checkpoint" } }),
      row("none", { epochs: null }),
    ];
    expect(names(sortRuns(rs, "epochs", "desc"))).toEqual(["old", "trained", "none"]);
  });

  it("updated sorts by time, newest first when descending", () => {
    const at = (iso: string | null) => ({ bestPt: true, lastPt: false, resultsJson: true, resultsInferenceJson: false, wandbIdTxt: false, newestMtime: iso });
    const rs = [row("old", { files: at("2026-01-01T00:00:00.000Z") }), row("new", { files: at("2026-10-01T00:00:00.000Z") }), row("never", { files: at(null) })];
    expect(names(sortRuns(rs, "updated", "desc"))).toEqual(["new", "old", "never"]);
    expect(names(sortRuns(rs, "updated", "asc"))).toEqual(["old", "new", "never"]);
  });

  it("names sort with numbers in numeric order, so v2 comes before v10", () => {
    const rs = ["r-v10-fold1", "r-v2-fold1", "r-v1-fold1"].map((n) => row(n));
    expect(names(sortRuns(rs, "name", "asc"))).toEqual(["r-v1-fold1", "r-v2-fold1", "r-v10-fold1"]);
    expect(names(sortRuns(rs, "name", "desc"))).toEqual(["r-v10-fold1", "r-v2-fold1", "r-v1-fold1"]);
  });

  it("ties are broken by name in ascending order, whatever the direction", () => {
    const rs = [withCross("b", 0.5), withCross("c", 0.5), withCross("a", 0.5)];
    expect(names(sortRuns(rs, "crossAp", "desc"))).toEqual(["a", "b", "c"]);
    expect(names(sortRuns(rs, "crossAp", "asc"))).toEqual(["a", "b", "c"]);
  });

  it("the order is the same whatever order the rows arrive in, even for names that differ only by case or leading zeros", () => {
    const x = [row("Run-A"), row("run-a"), row("r-v02-fold1"), row("r-v2-fold1")];
    const y = [...x].reverse();
    for (const key of SORT_KEYS) for (const dir of ["asc", "desc"] as const) expect(names(sortRuns(x, key, dir))).toEqual(names(sortRuns(y, key, dir)));
  });

  it("does not change the order of its input", () => {
    const rs = [withCross("a", 0.1), withCross("b", 0.9)];
    sortRuns(rs, "crossAp", "desc");
    expect(names(rs)).toEqual(["a", "b"]);
  });
});

describe("defaultDirection", () => {
  it("text columns start ascending; numbers and times start with the biggest or newest", () => {
    const expected: Record<SortKey, "asc" | "desc"> = {
      name: "asc", status: "asc", detector: "asc", crossAp: "desc", crossF1: "desc", inDomainAp: "desc", epochs: "desc", updated: "desc",
    };
    expect(SORT_KEYS.slice().sort()).toEqual(Object.keys(expected).sort());
    for (const k of SORT_KEYS) expect(defaultDirection(k)).toBe(expected[k]);
  });
});
