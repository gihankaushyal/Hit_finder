import { describe, it, expect } from "vitest";
import { execFile } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { promisify } from "node:util";
import { FakeGh } from "./helpers/fakeGh";
import { main, USAGE } from "../server/cli/kanbanSync";
import type { SyncDeps, SyncState } from "../server/lib/kanbanSync";

const run = promisify(execFile);
const DASHBOARD = path.resolve(import.meta.dirname, "..");
const FIXTURE = path.join(DASHBOARD, "tests/fixtures/kanban-sample.md");
const TEXT = "## Open decisions\n\n- [x] **Decided**\n- [ ] Open one\n\n## In progress\n\n- [ ] Working\n";

function setup(opts: { file?: string | null; stateExists?: boolean } = {}) {
  const gh = new FakeGh();
  const h = {
    file: (opts.file === undefined ? TEXT : opts.file) as string | null,
    state: { labelsEnsured: false, items: {} } as SyncState,
    stateExists: opts.stateExists ?? false,
    fileWrites: 0,
    stateSaves: 0,
    out: [] as string[],
    err: [] as string[],
  };
  const sync: SyncDeps = {
    gh,
    readFile: () => h.file,
    writeFile: (t) => {
      h.file = t;
      h.fileWrites++;
    },
    loadState: () => structuredClone(h.state),
    saveState: (s) => {
      h.state = structuredClone(s);
      h.stateSaves++;
      h.stateExists = true;
    },
    sleep: async () => {},
  };
  const io = { sync, stateExists: () => h.stateExists, out: (l: string) => h.out.push(l), err: (l: string) => h.err.push(l) };
  return { gh, h, io, text: () => h.out.join("\n") };
}

describe("kanban:sync CLI main()", () => {
  it("--dry-run prints counts and one line per action, writes nothing and calls no gh write", async () => {
    const { gh, h, io, text } = setup();
    const code = await main(["--dry-run"], io);
    expect(code).toBe(0);
    expect(gh.writes).toEqual([]);
    expect(h.fileWrites).toBe(0);
    expect(h.stateSaves).toBe(0);
    const t = text();
    expect(t).toMatch(/to create: 2/);
    expect(t).toMatch(/to create and close: 1/);
    for (const label of ["to update", "to append", "to link", "conflicts", "skipped", "not in file"]) expect(t).toContain(label);
    expect(t).toMatch(/create and close: Decided/);
    expect(t).toMatch(/create: Open one/);
    expect(t).toMatch(/dry run/i);
  });

  it("--dry-run on an already-imported setup reports updates and appends", async () => {
    const { gh, io, text } = setup({ stateExists: true, file: "## A\n\n- [ ] one <!-- gh:#1 -->\n" });
    gh.seed({ title: "one", state: "CLOSED" });
    gh.seed({ title: "from github" });
    await main(["--dry-run"], io);
    expect(text()).toMatch(/to update: 1/);
    expect(text()).toMatch(/to append: 1/);
    expect(text()).toMatch(/tick #1/);
    expect(text()).toMatch(/append to Inbox: #2 from github/);
  });

  it("refuses a first import without --yes: prints the dry-run summary and how to proceed, changes nothing", async () => {
    const { gh, h, io, text } = setup({ stateExists: false });
    const code = await main([], io);
    expect(code).not.toBe(0);
    expect(gh.writes).toEqual([]);
    expect(h.fileWrites).toBe(0);
    expect(h.stateSaves).toBe(0);
    expect(text()).toMatch(/to create: 2/);
    expect(h.err.join("\n")).toMatch(/first import/i);
    expect(h.err.join("\n")).toMatch(/--yes/);
  });

  it("performs the first import with --yes and summarises what it did", async () => {
    const { gh, h, io, text } = setup({ stateExists: false });
    const code = await main(["--yes"], io);
    expect(code).toBe(0);
    expect(gh.count("create")).toBe(3);
    expect(h.file).toContain("<!-- gh:#1 -->");
    expect(text()).toMatch(/created: 2/);
    expect(text()).toMatch(/created and closed: 1/);
    expect(text()).not.toMatch(/dry run/i);
  });

  it("a normal run on an imported setup needs no --yes", async () => {
    const { gh, io } = setup({ stateExists: true });
    expect(await main([], io)).toBe(0);
    expect(gh.count("create")).toBe(3);
  });

  it("exits non-zero when the sync aborted", async () => {
    const { gh, h, io } = setup({ stateExists: true, file: "## A\n\n- [ ] one\n- [ ] two\n" });
    gh.onCreate = () => {
      h.file = h.file + "\n- [ ] edited meanwhile\n";
    };
    expect(await main([], io)).toBe(1);
    expect(h.err.join("\n")).toMatch(/aborted|stopped/i);
  });

  it("exits non-zero with a one-line message when gh fails", async () => {
    const { gh, h, io } = setup({ stateExists: true });
    gh.failCreateAt = 1;
    expect(await main([], io)).toBe(1);
    expect(h.err.filter((l) => /failed/i.test(l))).toHaveLength(1);
  });

  it("rejects unknown flags with the usage text", async () => {
    const { gh, h, io } = setup();
    expect(await main(["--dry-run", "--bogus"], io)).toBe(2);
    expect(h.err.join("\n")).toContain(USAGE);
    expect(gh.jsonCalls).toEqual([]);
  });

  it("--help prints usage and exits 0", async () => {
    const { h, io } = setup();
    expect(await main(["--help"], io)).toBe(0);
    expect(h.out.join("\n")).toContain(USAGE);
  });
});

describe("kanban:sync CLI spawned with a fake gh", () => {
  function sandbox(markdown: string) {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "kanban-cli-"));
    const md = path.join(dir, "kanban.md");
    fs.writeFileSync(md, markdown);
    const log = path.join(dir, "gh.log");
    const store = path.join(dir, "issues.json");
    fs.writeFileSync(store, "[]");
    const gh = path.join(dir, "fake-gh.mjs");
    fs.writeFileSync(
      gh,
      `#!/usr/bin/env node
import fs from "node:fs";
const args = process.argv.slice(2);
fs.appendFileSync(${JSON.stringify(log)}, JSON.stringify(args) + "\\n");
const store = JSON.parse(fs.readFileSync(${JSON.stringify(store)}, "utf8"));
if (args[0] === "issue" && args[1] === "list") { process.stdout.write(JSON.stringify(store)); process.exit(0); }
if (args[0] === "issue" && args[1] === "create") {
  const n = store.length + 1;
  const flag = (k) => (args.find((a) => a.startsWith("--" + k + "=")) || "").slice(k.length + 3);
  store.push({ number: n, title: flag("title"), state: "OPEN", labels: args.filter((a) => a.startsWith("--label=")).map((a) => ({ name: a.slice(8) })), body: flag("body"), url: "https://github.com/o/r/issues/" + n, updatedAt: "2026-10-07T00:00:00Z" });
  fs.writeFileSync(${JSON.stringify(store)}, JSON.stringify(store));
  process.stdout.write("https://github.com/o/r/issues/" + n + "\\n");
  process.exit(0);
}
process.exit(0);
`,
      { mode: 0o755 },
    );
    const env = {
      ...process.env,
      DASH_GH_BIN: gh,
      DASH_KANBAN: md,
      DASH_STATE_DIR: path.join(dir, "state"),
      DASH_REPO_ROOT: dir,
    };
    const cli = (...args: string[]) =>
      run(path.join(DASHBOARD, "node_modules/.bin/tsx"), ["server/cli/kanbanSync.ts", ...args], { cwd: DASHBOARD, env }).then(
        (r) => ({ code: 0, stdout: r.stdout, stderr: r.stderr }),
        (e) => ({ code: e.code as number, stdout: e.stdout as string, stderr: e.stderr as string }),
      );
    const ghCalls = () => (fs.existsSync(log) ? fs.readFileSync(log, "utf8").trim().split("\n").filter(Boolean).map((l) => JSON.parse(l) as string[]) : []);
    return { dir, md, state: path.join(dir, "state", "kanban-sync.json"), cli, ghCalls };
  }

  it("dry run on a copy of the fixture: summary, no gh write, file and state untouched", async () => {
    const s = sandbox(fs.readFileSync(FIXTURE, "utf8"));
    const before = fs.readFileSync(s.md, "utf8");
    const r = await s.cli("--dry-run");
    expect(r.code).toBe(0);
    expect(r.stdout).toMatch(/to create: /);
    expect(s.ghCalls().every((c) => c[0] === "issue" && c[1] === "list")).toBe(true);
    expect(fs.readFileSync(s.md, "utf8")).toBe(before);
    expect(fs.existsSync(s.state)).toBe(false);
  }, 30_000);

  it("refuses the first real import without --yes", async () => {
    const s = sandbox(fs.readFileSync(FIXTURE, "utf8"));
    const r = await s.cli();
    expect(r.code).not.toBe(0);
    expect(r.stderr).toMatch(/--yes/);
    expect(s.ghCalls().every((c) => c[1] === "list")).toBe(true);
    expect(fs.existsSync(s.state)).toBe(false);
  }, 30_000);

  it("imports with --yes, then a second run is a no-op", async () => {
    const s = sandbox("## Open decisions\n\n- [ ] only one\n");
    const r = await s.cli("--yes");
    expect(r.code).toBe(0);
    expect(fs.readFileSync(s.md, "utf8")).toContain("<!-- gh:#1 -->");
    expect(fs.existsSync(s.state)).toBe(true);
    const creates = s.ghCalls().filter((c) => c[0] === "issue" && c[1] === "create").length;
    expect(creates).toBe(1);
    const r2 = await s.cli();
    expect(r2.code).toBe(0);
    expect(s.ghCalls().filter((c) => c[0] === "issue" && c[1] === "create")).toHaveLength(1);
  }, 30_000);
});
