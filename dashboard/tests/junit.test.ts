import { it, expect } from "vitest";
import { parseJunit } from "../server/lib/junit";

const xml = `<?xml version="1.0"?><testsuites><testsuite name="pytest" errors="1" failures="1" skipped="2" tests="10" time="12.5">
<testcase classname="tests.test_a" name="test_ok" time="0.1"/>
<testcase classname="tests.test_a" name="test_bad" time="0.2"><failure message="assert 1 == 2">trace</failure></testcase>
<testcase classname="tests.test_b" name="test_err" time="0.2"><error message="boom">trace</error></testcase>
</testsuite></testsuites>`;

it("summarises counts and failures", () => {
  const s = parseJunit(xml);
  expect(s).toMatchObject({ tests: 10, failed: 1, errors: 1, skipped: 2, passed: 6, durationSec: 12.5 });
  expect(s.failures).toEqual([
    { name: "tests.test_a::test_bad", message: "assert 1 == 2" },
    { name: "tests.test_b::test_err", message: "boom" },
  ]);
});
it("handles a single testsuite root and a single testcase", () => {
  const s = parseJunit(`<testsuite tests="1" failures="0" errors="0" skipped="0" time="0.5"><testcase classname="c" name="n"/></testsuite>`);
  expect(s).toMatchObject({ tests: 1, passed: 1, failed: 0 });
  expect(s.failures).toEqual([]);
});
it("sums several testsuites", () => {
  const s = parseJunit(`<testsuites><testsuite tests="2" failures="0" errors="0" skipped="0" time="1"/><testsuite tests="3" failures="0" errors="0" skipped="1" time="2"/></testsuites>`);
  expect(s).toMatchObject({ tests: 5, passed: 4, skipped: 1, durationSec: 3 });
});
it("falls back to element text when message attribute is missing", () => {
  const s = parseJunit(`<testsuite tests="1" failures="1" errors="0" skipped="0" time="0"><testcase classname="c" name="n"><failure>plain text</failure></testcase></testsuite>`);
  expect(s.failures).toEqual([{ name: "c::n", message: "plain text" }]);
});
it("throws on non-junit input", () => {
  expect(() => parseJunit("<html/>")).toThrow(/junit/i);
});
it("throws on malformed XML", () => {
  expect(() => parseJunit("<testsuite><testcase")).toThrow(/junit/i);
  expect(() => parseJunit("")).toThrow(/junit/i);
});
