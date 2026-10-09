import { XMLParser, XMLValidator } from "fast-xml-parser";
import type { JunitSummary } from "../../shared/types";

const MAX_MESSAGE_CHARS = 500;
const ARRAY_TAGS = new Set(["testsuite", "testcase", "failure", "error"]);

type Obj = Record<string, unknown>;

const parser = new XMLParser({
  ignoreAttributes: false,
  attributeNamePrefix: "@_",
  parseAttributeValue: false,
  isArray: (name) => ARRAY_TAGS.has(name),
});

const num = (v: unknown): number => {
  const n = Number(v);
  return Number.isFinite(n) ? n : 0;
};

function messageOf(node: unknown): string {
  let msg = "";
  if (typeof node === "string") msg = node;
  else if (node && typeof node === "object") {
    const o = node as Obj;
    msg = typeof o["@_message"] === "string" && o["@_message"] ? (o["@_message"] as string) : typeof o["#text"] === "string" ? (o["#text"] as string) : "";
  }
  return msg.slice(0, MAX_MESSAGE_CHARS);
}

export function parseJunit(xml: string): JunitSummary {
  if (XMLValidator.validate(xml) !== true) throw new Error("invalid junit XML");
  const doc = parser.parse(xml) as Obj;
  const root = (doc.testsuites ?? doc) as Obj;
  const suites = (root && typeof root === "object" ? (root.testsuite as Obj[] | undefined) : undefined) ?? [];
  if (suites.length === 0 && !(doc.testsuites && typeof doc.testsuites === "object")) {
    throw new Error("not a junit report: no testsuite element");
  }

  const out: JunitSummary = { tests: 0, passed: 0, failed: 0, errors: 0, skipped: 0, durationSec: 0, failures: [] };
  for (const s of suites) {
    out.tests += num(s["@_tests"]);
    out.failed += num(s["@_failures"]);
    out.errors += num(s["@_errors"]);
    out.skipped += num(s["@_skipped"]);
    out.durationSec += num(s["@_time"]);
    for (const tc of (s.testcase as Obj[] | undefined) ?? []) {
      const bad = [...((tc.failure as unknown[] | undefined) ?? []), ...((tc.error as unknown[] | undefined) ?? [])];
      if (bad.length > 0) {
        out.failures.push({ name: `${String(tc["@_classname"] ?? "")}::${String(tc["@_name"] ?? "")}`, message: messageOf(bad[0]) });
      }
    }
  }
  out.passed = Math.max(0, out.tests - out.failed - out.errors - out.skipped);
  out.durationSec = Math.round(out.durationSec * 1000) / 1000;
  return out;
}
