import { describe, it, expect } from "vitest";
import { dropSeen, reconcile, type LogLine } from "../../web/src/lib/logLines";

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

  it("starts over, with new ids, when nothing lines up", () => {
    const out = reconcile(lines("a", "b"), ["x", "y"], counter());
    expect(texts(out)).toEqual(["x", "y"]);
    expect(out.map((l) => l.id)).toEqual([100, 101]);
  });

  it("builds a snapshot from nothing", () => {
    expect(texts(reconcile([], ["a", "b"], counter()))).toEqual(["a", "b"]);
  });
});

describe("dropSeen", () => {
  it("keeps buffered lines the snapshot does not hold yet", () => {
    expect(texts(dropSeen(["a", "b"], lines("c", "d")))).toEqual(["c", "d"]);
  });

  it("drops buffered lines the snapshot already ends with", () => {
    expect(texts(dropSeen(["a", "b", "c"], lines("b", "c", "d")))).toEqual(["d"]);
  });

  it("drops everything when the snapshot has caught up", () => {
    expect(dropSeen(["a", "b"], lines("a", "b"))).toEqual([]);
  });

  it("leaves an empty buffer alone", () => {
    expect(dropSeen(["a"], [])).toEqual([]);
  });
});
