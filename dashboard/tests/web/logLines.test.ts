import { describe, it, expect } from "vitest";
import { reconcile, type LogLine } from "../../web/src/lib/logLines";

function counter(start = 100) {
  let n = start;
  return () => n++;
}
const lines = (...texts: string[]): LogLine[] => texts.map((text, id) => ({ id, text }));
const texts = (l: LogLine[]) => l.map((x) => x.text);

describe("reconcile", () => {
  it("keeps every id when the snapshot is the same as what is shown", () => {
    const prev = lines("a", "b", "c");
    const out = reconcile(prev, ["a", "b", "c"], counter());
    expect(out.map((l) => l.id)).toEqual([0, 1, 2]);
  });

  it("keeps ids for rows that survive when the window slides, and gives new rows new ids", () => {
    const prev = lines("a", "b", "c");
    const out = reconcile(prev, ["b", "c", "d"], counter());
    expect(texts(out)).toEqual(["b", "c", "d"]);
    expect(out.map((l) => l.id)).toEqual([1, 2, 100]);
  });

  it("appends lines the snapshot has beyond what is shown", () => {
    const out = reconcile(lines("a"), ["a", "b", "c"], counter());
    expect(texts(out)).toEqual(["a", "b", "c"]);
    expect(out.map((l) => l.id)).toEqual([0, 100, 101]);
  });

  it("ignores a snapshot that lags the stream: lines already shown are not removed", () => {
    const prev = lines("a", "b", "c", "d", "e");
    expect(reconcile(prev, ["a", "b", "c"], counter())).toBe(prev);
    expect(reconcile(prev, ["b", "c", "d"], counter())).toBe(prev);
  });

  it("starts over, with new ids, when nothing lines up (a different run)", () => {
    const out = reconcile(lines("a", "b"), ["x", "y"], counter());
    expect(texts(out)).toEqual(["x", "y"]);
    expect(out.map((l) => l.id)).toEqual([100, 101]);
  });

  it("clears the output for an empty snapshot", () => {
    expect(reconcile(lines("a", "b"), [], counter())).toEqual([]);
  });

  it("builds a snapshot from nothing", () => {
    expect(texts(reconcile([], ["a", "b"], counter()))).toEqual(["a", "b"]);
  });
});
