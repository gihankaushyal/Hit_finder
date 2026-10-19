import { it, expect } from "vitest";
import { parsePrBody } from "../server/lib/prBody";
const body = [
  "## Summary",
  "- `train_asymmetric.py` now requires `--run-name-prefix`",
  "- widen results regex",
  "  continued line",
  "",
  "## Test plan",
  "- [x] pytest tests/ -v",
  "- [ ] smoke run on ePix10k",
  "",
  "🤖 Generated with Claude Code",
].join("\n");
it("extracts summary bullets and test plan", () => {
  const r = parsePrBody(body);
  expect(r.summary).toEqual([
    "`train_asymmetric.py` now requires `--run-name-prefix`",
    "widen results regex continued line",
  ]);
  expect(r.testPlan).toEqual([
    { text: "pytest tests/ -v", checked: true },
    { text: "smoke run on ePix10k", checked: false },
  ]);
});
it("falls back to the first paragraph when there is no Summary heading", () => {
  expect(parsePrBody("Just a sentence.\n\nMore.").summary).toEqual(["Just a sentence."]);
  expect(parsePrBody("").summary).toEqual([]);
  expect(parsePrBody("").testPlan).toEqual([]);
});
it("matches headings case-insensitively and tolerates CRLF", () => {
  expect(parsePrBody("## SUMMARY\r\n- a\r\n").summary).toEqual(["a"]);
});
it("ignores headings and checklist items inside fenced code blocks", () => {
  const b = [
    "## Summary",
    "- real bullet",
    "```bash",
    "# comment that is not a heading",
    "- fake bullet",
    "```",
    "~~~",
    "## Test plan",
    "- [x] fake check",
    "~~~",
    "",
    "## Test plan",
    "- [ ] real check",
  ].join("\n");
  const r = parsePrBody(b);
  expect(r.summary).toEqual(["real bullet"]);
  expect(r.testPlan).toEqual([{ text: "real check", checked: false }]);
});
