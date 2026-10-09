#!/usr/bin/env node
// Stand-in for the GitHub CLI, used only by the end-to-end tests (DASH_GH_BIN points here).
// It implements just the subcommands the dashboard server uses. Anything else exits 2 with a clear
// message on stderr, so an unexpected call fails the test instead of passing silently.
//
// Environment:
//   FAKE_GH_STATE  path of the JSON file holding the issue store (required)
//   FAKE_GH_CALLS  path of a JSON-lines file; one line is appended per invocation (required)
//   FAKE_GH_SEED   "default" (default) or "empty": what the store holds when first created
// Failure injection: if `${FAKE_GH_STATE}.fail` exists and holds a JSON array of command prefixes such as
// ["pr list", "issue list"], a matching command exits 1 with "fake-gh: injected failure" on stderr.
import fs from "node:fs";

const STATE = process.env.FAKE_GH_STATE;
const CALLS = process.env.FAKE_GH_CALLS;
const REPO_URL = "https://github.com/example/hit-finder";
const argv = process.argv.slice(2);

function die(code, message) {
  process.stderr.write(`fake-gh: ${message}\n`);
  process.exit(code);
}
if (!STATE || !CALLS) die(2, "FAKE_GH_STATE and FAKE_GH_CALLS must be set");

fs.appendFileSync(CALLS, JSON.stringify({ args: argv, at: new Date().toISOString() }) + "\n");

const iso = (minutesAgo) => new Date(Date.now() - minutesAgo * 60_000).toISOString();

function seed() {
  if (process.env.FAKE_GH_SEED === "empty") return { nextIssue: 1, issues: [], prs: [], runs: [] };
  return {
    nextIssue: 2,
    issues: [
      { number: 1, title: "Eiger4M geometry looks off", state: "OPEN", labels: ["bug"], body: "", createdAt: iso(600), updatedAt: iso(600) },
    ],
    prs: [
      {
        number: 41, title: "Add frame cache", state: "MERGED", mergedAt: iso(3000), headRefName: "phase-05-frame-cache", baseRefName: "main",
        additions: 220, deletions: 14, changedFiles: 3, body: "## Summary\n- Caches assembled frames\n\n## Test plan\n- [x] pytest\n",
        files: [{ path: "src/data/frame_cache.py", additions: 200, deletions: 10 }], statusCheckRollup: [],
      },
      {
        number: 42, title: "Add the status dashboard", state: "MERGED", mergedAt: iso(120), headRefName: "phase-05-dashboard", baseRefName: "main",
        additions: 1200, deletions: 40, changedFiles: 4,
        body: "## Summary\n- Shows the latest pull request\n- Syncs the kanban file with GitHub issues\n\n## Test plan\n- [x] npm test\n- [ ] smoke on Sol\n",
        files: [
          { path: "dashboard/server/main.ts", additions: 70, deletions: 0 },
          { path: "dashboard/web/src/App.tsx", additions: 40, deletions: 2 },
        ],
        statusCheckRollup: [],
      },
      {
        number: 43, title: "Tune the hit finder", state: "OPEN", mergedAt: null, headRefName: "phase-05-tune", baseRefName: "main",
        additions: 12, deletions: 3, changedFiles: 1, body: "", files: [],
        statusCheckRollup: [
          { __typename: "CheckRun", name: "lint", status: "COMPLETED", conclusion: "SUCCESS" },
          { __typename: "CheckRun", name: "unit", status: "COMPLETED", conclusion: "FAILURE" },
          { __typename: "StatusContext", context: "ci/other", state: "SUCCESS" },
        ],
      },
    ],
    runs: [
      { databaseId: 9002, name: "CI", status: "completed", conclusion: "success", headBranch: "main", event: "push", createdAt: iso(60) },
      { databaseId: 9001, name: "CI", status: "completed", conclusion: "failure", headBranch: "phase-05-tune", event: "push", createdAt: iso(300) },
    ],
  };
}

function load() {
  try {
    return JSON.parse(fs.readFileSync(STATE, "utf8"));
  } catch (err) {
    if (err.code !== "ENOENT") throw err;
    return null;
  }
}
function save(state) {
  const tmp = `${STATE}.${process.pid}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(state, null, 2));
  fs.renameSync(tmp, STATE);
}

// Several gh processes can run at once; mutations take a lock directory so none is lost.
function withState(mutating, fn) {
  const lock = `${STATE}.lock`;
  const deadline = Date.now() + 10_000;
  for (;;) {
    try {
      fs.mkdirSync(lock);
      break;
    } catch (err) {
      if (err.code !== "EEXIST") throw err;
      if (Date.now() > deadline) die(1, "could not take the state lock");
      Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 15);
    }
  }
  try {
    let state = load();
    if (!state) {
      state = seed();
      save(state);
    }
    const result = fn(state);
    if (mutating) save(state);
    return result;
  } finally {
    fs.rmdirSync(lock);
  }
}

// ---- argument handling ----
const positional = [];
const flags = new Map(); // "--name" -> string[] of values (always the `--name=value` form, as the server sends)
for (const a of argv) {
  const m = /^--([a-z-]+)(?:=([\s\S]*))?$/.exec(a);
  if (m) flags.set(m[1], [...(flags.get(m[1]) ?? []), m[2] ?? ""]);
  else positional.push(a);
}
// The server passes some flags as separate words (`--label task`, `--state all`, `--limit 40`, `--json a,b`).
// Re-read the raw argv for those so both spellings work.
function valueOf(name) {
  if (flags.has(name) && flags.get(name)[0] !== "") return flags.get(name)[0];
  const i = argv.indexOf(`--${name}`);
  return i >= 0 ? argv[i + 1] : undefined;
}
const SPACED = new Set(["label", "state", "limit", "json"]);
const words = [];
for (let i = 0; i < argv.length; i++) {
  const a = argv[i];
  if (a.startsWith("--")) {
    if (!a.includes("=") && SPACED.has(a.slice(2))) i++; // skip its value
    continue;
  }
  words.push(a);
}

const command = `${words[0] ?? ""} ${words[1] ?? ""}`.trim();
try {
  const inject = JSON.parse(fs.readFileSync(`${STATE}.fail`, "utf8"));
  if (Array.isArray(inject) && inject.some((p) => command.startsWith(p))) die(1, `injected failure for "${command}"`);
} catch (err) {
  if (err.code !== "ENOENT") throw err;
}

const pick = (obj, fields) => Object.fromEntries(fields.filter((f) => f in obj).map((f) => [f, obj[f]]));
const jsonFields = () => (valueOf("json") ?? "").split(",").filter(Boolean);
const print = (v) => process.stdout.write(JSON.stringify(v) + "\n");
const issueUrl = (n) => `${REPO_URL}/issues/${n}`;
const asIssue = (i) => ({ ...i, url: issueUrl(i.number), labels: i.labels.map((name) => ({ name })) });
const nowIso = () => new Date().toISOString();
const flag = (name) => flags.get(name)?.[0];

switch (command) {
  case "pr list": {
    const fields = jsonFields();
    withState(false, (s) => print(s.prs.map((p) => pick({ ...p, url: `${REPO_URL}/pull/${p.number}` }, fields))));
    break;
  }
  case "pr view": {
    const n = Number(words[2]);
    const fields = jsonFields();
    withState(false, (s) => {
      const p = s.prs.find((x) => x.number === n);
      if (!p) die(1, `no pull request #${n}`);
      print(pick({ ...p, url: `${REPO_URL}/pull/${p.number}` }, fields));
    });
    break;
  }
  case "issue list": {
    const label = valueOf("label");
    const state = valueOf("state") ?? "open";
    const exclude = /^-label:(.+)$/.exec(flag("search") ?? "")?.[1];
    const fields = jsonFields();
    withState(false, (s) => {
      const out = s.issues
        .filter((i) => state === "all" || i.state.toLowerCase() === state)
        .filter((i) => !label || i.labels.includes(label))
        .filter((i) => !exclude || !i.labels.includes(exclude))
        .map((i) => pick(asIssue(i), fields));
      print(out);
    });
    break;
  }
  case "issue create": {
    const title = flag("title");
    if (title === undefined) die(2, "issue create needs --title=");
    withState(true, (s) => {
      const n = s.nextIssue++;
      s.issues.push({ number: n, title, state: "OPEN", labels: flags.get("label") ?? [], body: flag("body") ?? "", createdAt: nowIso(), updatedAt: nowIso() });
      process.stdout.write(`${issueUrl(n)}\n`);
    });
    break;
  }
  case "issue close":
  case "issue reopen":
  case "issue edit": {
    const n = Number(words[2]);
    withState(true, (s) => {
      const issue = s.issues.find((i) => i.number === n);
      if (!issue) die(1, `no issue #${n}`);
      if (command === "issue close") issue.state = "CLOSED";
      else if (command === "issue reopen") issue.state = "OPEN";
      else {
        if (flag("body") !== undefined) issue.body = flag("body");
        const remove = flags.get("remove-label") ?? [];
        issue.labels = issue.labels.filter((l) => !remove.includes(l));
        for (const l of flags.get("add-label") ?? []) if (!issue.labels.includes(l)) issue.labels.push(l);
      }
      issue.updatedAt = nowIso();
    });
    break;
  }
  case "label create":
    if (!words[2]) die(2, "label create needs a name");
    break; // labels are not modelled
  case "run list": {
    const fields = jsonFields();
    withState(false, (s) => print(s.runs.map((r) => pick({ ...r, url: `${REPO_URL}/actions/runs/${r.databaseId}` }, fields))));
    break;
  }
  default:
    die(2, `unsupported command: gh ${argv.join(" ")}`);
}
