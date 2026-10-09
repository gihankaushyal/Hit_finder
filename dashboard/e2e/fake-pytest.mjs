#!/usr/bin/env node
// Stand-in for `python -m pytest`, used only by the end-to-end tests (DASH_PYTHON points here).
// The server calls it as: fake-pytest.mjs -m pytest tests/ -q --junitxml=<path>
// It prints two progress lines, writes a junit report with 5 tests and 1 failure, and exits 1.
//
// Environment:
//   FAKE_PYTEST_CALLS     path of a file; one line is appended per run (optional)
//   FAKE_PYTEST_DELAY_MS  pause between the lines (default 700), so a run is visibly "running"
import fs from "node:fs";

const arg = process.argv.slice(2).find((a) => a.startsWith("--junitxml="));
if (!arg) {
  process.stderr.write("fake-pytest: expected --junitxml=<path>\n");
  process.exit(2);
}
const junit = arg.slice("--junitxml=".length);
if (process.env.FAKE_PYTEST_CALLS) fs.appendFileSync(process.env.FAKE_PYTEST_CALLS, `${new Date().toISOString()} ${process.argv.slice(2).join(" ")}\n`);
const delay = Number(process.env.FAKE_PYTEST_DELAY_MS ?? 700);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

console.log("collected 5 items");
await sleep(delay);
console.log("tests/test_hitfinder.py ..F..");
await sleep(delay);

const xml = `<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" errors="0" failures="1" skipped="0" tests="5" time="1.250">
<testcase classname="tests.test_hitfinder" name="test_pass_one" time="0.1"/>
<testcase classname="tests.test_hitfinder" name="test_pass_two" time="0.1"/>
<testcase classname="tests.test_hitfinder" name="test_vote_aggregation_breaks_ties" time="0.3"><failure message="assert 0.4 == 0.5">AssertionError: assert 0.4 == 0.5</failure></testcase>
<testcase classname="tests.test_hitfinder" name="test_pass_three" time="0.1"/>
<testcase classname="tests.test_hitfinder" name="test_pass_four" time="0.1"/>
</testsuite></testsuites>
`;
fs.writeFileSync(junit, xml);
console.log("1 failed, 4 passed in 1.25s");
process.exit(1);
