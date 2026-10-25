import { describe, it, expect } from "vitest";
import { parseFrontmatter } from "../server/skills/frontmatter";

const wrap = (body: string) => `---\n${body}\n---\nbody text\nname: not-this\n`;

describe("parseFrontmatter", () => {
  it("reads plain values and ignores the body", () => {
    expect(parseFrontmatter(wrap("name: slurm-job\ndescription: Submit a job"))).toEqual({
      name: "slurm-job",
      description: "Submit a job",
    });
  });
  it("unquotes values and keeps colons and # inside quotes", () => {
    expect(parseFrontmatter(wrap(`a: "x: y # z"\nb: 'it''s'\nc: "say \\"hi\\""`))).toEqual({ a: "x: y # z", b: "it's", c: 'say "hi"' });
  });
  it("drops a trailing comment on a plain value", () => {
    expect(parseFrontmatter(wrap("name: alpha # the first"))).toEqual({ name: "alpha" });
  });
  it("joins folded and literal block scalars and indented continuations", () => {
    const out = parseFrontmatter(wrap("a: >\n  one\n  two\nb: |\n  one\n  two\nc: start\n  more\nd: next"));
    expect(out).toEqual({ a: "one two", b: "one\ntwo", c: "start more", d: "next" });
  });
  it("returns nothing without a header", () => {
    expect(parseFrontmatter("no header here")).toEqual({});
  });
  it("skips nested keys, lists, numbers and booleans", () => {
    expect(parseFrontmatter(wrap("a: [1, 2]\nb:\n  c: d\nn: 3\nflag: true\nname: ok"))).toEqual({ name: "ok" });
  });
  it("throws on a header that is not valid YAML", () => {
    expect(() => parseFrontmatter(wrap("name: [unclosed"))).toThrow();
  });
  it("reads a header after a byte-order mark, and an empty header", () => {
    expect(parseFrontmatter("\uFEFF---\nname: ok\n---\n")).toEqual({ name: "ok" });
    expect(parseFrontmatter("---\n---\nbody")).toEqual({});
  });
  it("closes the header only on a line that is exactly ---", () => {
    expect(parseFrontmatter("---\nname: ok\nnote: a\n  ----\nafter: yes\n---\n")).toMatchObject({ name: "ok" });
    expect(parseFrontmatter("---\nname: no closing fence\n")).toEqual({});
  });
  it("keeps a quoted true as text", () => {
    expect(parseFrontmatter(wrap('a: "true"'))).toEqual({ a: "true" });
  });
});
