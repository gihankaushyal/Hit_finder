import { describe, it, expect, vi } from "vitest";
import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { FakeGh } from "./helpers/fakeGh";
import {
  runSync, listTasks, labelsForSection, fileSyncDeps, TASK_LABELS, type SyncDeps, type SyncState,
} from "../server/lib/kanbanSync";
import type { Config } from "../server/config";

const THREE = [
  "## Open decisions (need a call)",
  "",
  "- [x] **Decided thing** — done already",
  "- [ ] **Open thing**",
  "",
  "## In progress (Phase 5)",
  "",
  "- [ ] Working on it",
  "",
].join("\n");

function mk(text: string | null, gh = new FakeGh()) {
  const h = {
    gh,
    file: text as string | null,
    state: { labelsEnsured: false, items: {} } as SyncState,
    fileWrites: 0,
    stateSaves: 0,
    sleeps: 0,
    onSleep: () => {},
  };
  const deps: SyncDeps = {
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
    },
    sleep: async () => {
      h.sleeps++;
      h.onSleep();
    },
  };
  return { h, deps, gh };
}
const sync = (deps: SyncDeps, dryRun = false) => runSync(deps, { dryRun });

describe("labelsForSection", () => {
  it("maps headings to labels", () => {
    expect(labelsForSection("In progress (Phase 5)")).toEqual(["task", "status:in-progress", "kind:tracked"]);
    expect(labelsForSection("Open decisions (need a call)")).toEqual(["task", "status:todo", "kind:decision"]);
    expect(labelsForSection("Awaiting merge (PRs open)")).toEqual(["task", "status:todo", "kind:awaiting-merge"]);
    expect(labelsForSection("Explorable objectives")).toEqual(["task", "status:todo", "kind:explore"]);
    expect(labelsForSection("Inbox (added via dashboard)")).toEqual(["task", "status:todo"]);
    expect(labelsForSection("Whatever")).toEqual(["task", "status:todo", "kind:tracked"]);
  });
  it("has the eight task labels", () => {
    expect(TASK_LABELS.map((l) => l.name)).toEqual([
      "task", "status:todo", "status:in-progress", "status:blocked",
      "kind:decision", "kind:explore", "kind:awaiting-merge", "kind:tracked",
    ]);
  });
});

describe("listTasks", () => {
  it("asks gh for task issues in every state and maps labels to names", async () => {
    const gh = new FakeGh();
    gh.seed({ title: "a", labels: ["task", "x"] });
    const tasks = await listTasks(gh);
    expect(gh.jsonCalls[0]).toEqual([
      "issue", "list", "--label", "task", "--state", "all", "--limit", "1000",
      "--json", "number,title,state,labels,body,url,updatedAt",
    ]);
    expect(tasks[0].labels).toEqual(["task", "x"]);
  });
});

describe("runSync", () => {
  it("1. first sync creates issues with labels, closes the checked one, writes markers, ensures labels once", async () => {
    const { h, deps, gh } = mk(THREE);
    const r = await sync(deps);
    expect([...gh.issues.values()].map((i) => [i.number, i.title, i.state, i.labels])).toEqual([
      [1, "Decided thing", "CLOSED", ["task", "status:todo", "kind:decision"]],
      [2, "Open thing", "OPEN", ["task", "status:todo", "kind:decision"]],
      [3, "Working on it", "OPEN", ["task", "status:in-progress", "kind:tracked"]],
    ]);
    expect(gh.writes.filter((w) => w[0] === "label")).toHaveLength(8);
    expect(h.file).toContain("- [x] **Decided thing** — done already <!-- gh:#1 -->");
    expect(h.file).toContain("- [ ] **Open thing** <!-- gh:#2 -->");
    expect(h.file).toContain("- [ ] Working on it <!-- gh:#3 -->");
    expect(h.sleeps).toBe(2);
    expect(gh.issues.get(1)!.body).toContain("_Synced from phase-05-kanban.md, section: Open decisions (need a call)_");
    expect(r.actions.map((a) => a.type)).toEqual(["create", "create", "create"]);
    expect(h.state.labelsEnsured).toBe(true);
  });

  it("2. a second sync makes zero write calls, leaves the file byte-identical and saves nothing", async () => {
    const { h, deps, gh } = mk(THREE);
    await sync(deps);
    const fileAfter = h.file;
    const writes = gh.writes.length;
    const saves = h.stateSaves;
    const fileWrites = h.fileWrites;
    const r = await sync(deps);
    expect(gh.writes).toHaveLength(writes);
    expect(h.file).toBe(fileAfter);
    expect(h.fileWrites).toBe(fileWrites);
    expect(h.stateSaves).toBe(saves);
    expect(r.actions).toEqual([]);
    expect(r.conflicts).toEqual([]);
  });

  it("3. dry run reports the creates and changes nothing; its actions match the real run", async () => {
    const { h, deps, gh } = mk(THREE);
    const dry = await sync(deps, true);
    expect(dry.actions).toEqual([
      { type: "create", title: "Decided thing", labels: ["task", "status:todo", "kind:decision"], closed: true },
      { type: "create", title: "Open thing", labels: ["task", "status:todo", "kind:decision"], closed: false },
      { type: "create", title: "Working on it", labels: ["task", "status:in-progress", "kind:tracked"], closed: false },
    ]);
    expect(gh.writes).toHaveLength(0);
    expect(h.file).toBe(THREE);
    expect(h.fileWrites).toBe(0);
    expect(h.stateSaves).toBe(0);
    expect(h.sleeps).toBe(0);
    const real = await sync(deps);
    expect(real.actions).toEqual(dry.actions);
  });

  it("4. ticking a box in markdown closes the issue; unticking reopens", async () => {
    const { h, deps, gh } = mk(THREE);
    await sync(deps);
    h.file = h.file!.replace("- [ ] **Open thing**", "- [x] **Open thing**");
    const r1 = await sync(deps);
    expect(r1.actions).toEqual([{ type: "close", issue: 2 }]);
    expect(gh.issues.get(2)!.state).toBe("CLOSED");
    h.file = h.file!.replace("- [x] **Open thing**", "- [ ] **Open thing**");
    const r2 = await sync(deps);
    expect(r2.actions).toEqual([{ type: "reopen", issue: 2 }]);
    expect(gh.issues.get(2)!.state).toBe("OPEN");
    expect((await sync(deps)).actions).toEqual([]);
  });

  it("5. closing on GitHub flips the checkbox and changes exactly one character", async () => {
    const { h, deps, gh } = mk(THREE);
    await sync(deps);
    const before = h.file!;
    gh.issues.get(3)!.state = "CLOSED";
    const r = await sync(deps);
    expect(r.actions).toEqual([{ type: "md-check", issue: 3, checked: true }]);
    let diff = 0;
    expect(h.file!.length).toBe(before.length);
    for (let i = 0; i < before.length; i++) if (before[i] !== h.file![i]) diff++;
    expect(diff).toBe(1);
    expect(h.file).toContain("- [x] Working on it");
  });

  it("6. a change on both sides keeps GitHub's state and reports a conflict", async () => {
    // With a consistent last-synced state both sides cannot disagree after both changing, so craft an
    // inconsistent record (as left by an interrupted run): last = checked, open.
    const { h, deps, gh } = mk(THREE);
    await sync(deps);
    h.state.items["3"] = { ...h.state.items["3"], checked: true, closed: false };
    gh.issues.get(3)!.state = "CLOSED"; // GitHub: open -> closed
    // markdown: checked -> unchecked (it is unchecked in the file already)
    const r = await sync(deps);
    expect(r.conflicts).toContain("#3 changed in both places; kept GitHub state");
    expect(r.actions).toContainEqual({ type: "md-check", issue: 3, checked: true });
    expect(h.file).toContain("- [x] Working on it");
    expect(gh.issues.get(3)!.state).toBe("CLOSED");
    expect(gh.count("reopen")).toBe(0);
  });

  it("7. editing item text updates the issue body exactly once", async () => {
    const { h, deps, gh } = mk(THREE);
    await sync(deps);
    h.file = h.file!.replace("- [ ] Working on it", "- [ ] Working on it harder");
    const r = await sync(deps);
    expect(r.actions).toEqual([{ type: "update-body", issue: 3 }]);
    expect(gh.issues.get(3)!.body).toContain("Working on it harder");
    expect(gh.issues.get(3)!.body).toContain("_Synced from phase-05-kanban.md");
    expect(gh.count("edit")).toBe(1);
    expect((await sync(deps)).actions).toEqual([]);
    expect(gh.count("edit")).toBe(1);
  });

  it("8. a new task issue is appended to the Inbox once", async () => {
    const { h, deps, gh } = mk(THREE);
    await sync(deps);
    gh.seed({ title: "Filed on GitHub" });
    const r = await sync(deps);
    expect(r.actions).toEqual([{ type: "md-append", issue: 4, title: "Filed on GitHub" }]);
    expect(h.file).toContain("## Inbox (added via dashboard)");
    expect(h.file).toContain("- [ ] Filed on GitHub <!-- gh:#4 -->");
    const after = h.file;
    expect((await sync(deps)).actions).toEqual([]);
    expect((await sync(deps)).actions).toEqual([]);
    expect(h.file).toBe(after);
    expect(gh.count("create")).toBe(3);
  });

  it("9. deleting a synced item reports it as notInFile and neither re-appends nor closes it", async () => {
    const { h, deps, gh } = mk(THREE);
    await sync(deps);
    h.file = h.file!.replace(/- \[ \] Working on it <!-- gh:#3 -->\n/, "");
    const closes = gh.count("close");
    const r = await sync(deps);
    expect(r.notInFile).toEqual([3]);
    expect(r.actions).toEqual([]);
    expect(h.file).not.toContain("Working on it");
    expect(gh.issues.get(3)!.state).toBe("OPEN");
    expect(gh.count("close")).toBe(closes);
    expect(gh.count("reopen")).toBe(0);
    const again = await sync(deps);
    expect(again.notInFile).toEqual([3]);
    expect(again.actions).toEqual([]);
  });

  it("10. a marker pointing at a missing issue is a conflict with no mutation", async () => {
    const { h, deps, gh } = mk("## S\n\n- [ ] Ghost <!-- gh:#99 -->\n");
    h.state.items["98"] = { checked: false, closed: false, bodyHash: "x" }; // state has records, so this is not a lost state file
    gh.next = 200;
    const r = await sync(deps);
    expect(r.conflicts).toEqual(["#99 is referenced in the file but was not found on GitHub"]);
    expect(r.actions).toEqual([]);
    expect(gh.writes.filter((w) => w[0] === "issue")).toHaveLength(0);
  });

  it("11. gh create output without an issue URL throws and leaves the file unwritten", async () => {
    const { h, deps, gh } = mk(THREE);
    gh.createOutput = () => "created something";
    await expect(sync(deps)).rejects.toThrow(/issue number/);
    expect(h.file).toBe(THREE);
    expect(h.fileWrites).toBe(0);
  });

  it("12. a title with quotes and $() is one unchanged argv element", async () => {
    const title = `a "quoted" $(rm -rf /) \`tick\' title`;
    const { deps, gh } = mk(`## S\n\n- [ ] ${title}\n`);
    await sync(deps);
    const create = gh.writes.find((w) => w[0] === "issue" && w[1] === "create")!;
    expect(create.filter((a) => a.startsWith("--title="))).toEqual([`--title=${title}`]);
    expect(create[0]).toBe("issue");
  });

  it("13. a missing kanban file does not throw and writes nothing", async () => {
    const { h, deps, gh } = mk(null);
    gh.seed({ title: "on github" });
    const r = await sync(deps);
    expect(r.actions).toEqual([]);
    expect(h.fileWrites).toBe(0);
    expect(h.file).toBeNull();
  });
});

const FIVE = ["## A", "", "- [ ] one", "- [ ] two", "- [ ] three", "- [x] four", "- [ ] five", ""].join("\n");

describe("runSync safety guarantees", () => {
  it("14. an interrupted run resumes at the next item instead of recreating earlier ones", async () => {
    const { h, deps, gh } = mk(FIVE);
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    gh.failCreateAt = 3;
    const r1 = await sync(deps); // a failed create is isolated: the run goes on with the next item
    spy.mockRestore();
    expect(r1.conflicts.join()).toMatch(/creating "three" failed/);
    expect(gh.issues.size).toBe(4);
    expect(h.file).toContain("- [ ] one <!-- gh:#1 -->");
    expect(h.file).toContain("- [ ] two <!-- gh:#2 -->");
    expect(h.file).not.toContain("three <!--");
    expect(h.state.importCompletedAt).toBeUndefined(); // a create is still left to do
    gh.failCreateAt = null;
    const r = await sync(deps);
    expect(r.actions.map((a) => (a.type === "create" ? a.title : a.type))).toEqual(["three"]);
    expect([...gh.issues.values()].map((i) => i.title).sort()).toEqual(["five", "four", "one", "three", "two"]);
    expect([...gh.issues.values()].find((i) => i.title === "four")!.state).toBe("CLOSED");
    expect(h.state.importCompletedAt).toBeDefined();
    expect((await sync(deps)).actions).toEqual([]);
  });

  it("15. the marker is on disk and state saved before the next create starts", async () => {
    const { h, deps, gh } = mk(FIVE);
    const seen: { markers: number; state: number }[] = [];
    const orig = gh.text.bind(gh);
    gh.text = async (args) => {
      if (args[0] === "issue" && args[1] === "create") {
        seen.push({ markers: (h.file!.match(/<!-- gh:#/g) ?? []).length, state: Object.keys(h.state.items).length });
      }
      return orig(args);
    };
    await sync(deps);
    expect(seen).toEqual([0, 1, 2, 3, 4].map((n) => ({ markers: n, state: n })));
  });

  it("16. a checked item is closed after creation; if the close fails the next run closes it, not re-creates", async () => {
    const { h, deps, gh } = mk("## A\n\n- [x] finished\n");
    gh.failClose = true;
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    const r1 = await sync(deps);
    spy.mockRestore();
    expect(r1.conflicts.join()).toMatch(/closing #1 after creating it failed/);
    expect(h.file).toContain("- [x] finished <!-- gh:#1 -->");
    expect(gh.issues.get(1)!.state).toBe("OPEN");
    gh.failClose = false;
    const r2 = await sync(deps);
    expect(r2.actions).toEqual([{ type: "close", issue: 1 }]);
    expect(gh.issues.get(1)!.state).toBe("CLOSED");
    expect(gh.count("create")).toBe(1);
    expect(h.file).toContain("- [x] finished");
    expect((await sync(deps)).actions).toEqual([]);
  });

  it("17. skips and reports an item whose title would be empty", async () => {
    const { deps, gh } = mk("## A\n\n- [ ] ** **\n- [ ] real\n");
    const r = await sync(deps);
    expect(gh.count("create")).toBe(1);
    expect(r.skipped).toEqual([{ title: "", section: "A", reason: "empty title" }]);
    const dry = await sync(deps, true);
    expect(dry.skipped).toHaveLength(1);
  });

  it("18. a title that looks like a flag is passed as a single --title= element", async () => {
    const { deps, gh } = mk("## A\n\n- [ ] --web\n");
    await sync(deps);
    const create = gh.writes.find((w) => w[0] === "issue" && w[1] === "create")!;
    expect(create).toContain("--title=--web");
    expect(create.some((a) => a === "--web")).toBe(false);
  });

  it("19. stops with a conflict if the file changes on disk during the sync, never overwriting the edit", async () => {
    const { h, deps, gh } = mk(FIVE);
    const edited = FIVE + "\n- [ ] user added this meanwhile\n";
    // Edit lands while the second create is pending, i.e. after the first marker was written.
    gh.onCreate = (n) => {
      if (n === 2) h.file = edited;
    };
    const r = await sync(deps);
    expect(r.aborted).toBe(true);
    expect(r.conflicts.join()).toMatch(/#2 was created on GitHub but the kanban file changed/);
    expect(h.file).toBe(edited);
    expect(gh.count("create")).toBe(2);
  });

  it("20. stops before creating anything when the file changes during the pause between creates", async () => {
    const { h, deps, gh } = mk(FIVE);
    h.onSleep = () => {
      h.file = h.file + "\n- [ ] user line\n";
    };
    const r = await sync(deps);
    expect(r.aborted).toBe(true);
    expect(gh.count("create")).toBe(1);
    expect(h.file).toContain("user line");
    expect(r.conflicts.join()).toMatch(/changed on disk/);
  });

  it("21. never closes or reopens issues just because items disappeared", async () => {
    const { h, deps, gh } = mk(FIVE);
    await sync(deps);
    const before = gh.writes.length;
    h.file = "# empty now\n";
    const r = await sync(deps);
    expect(r.notInFile).toEqual([1, 2, 3, 4, 5]);
    expect(gh.writes).toHaveLength(before);
  });

  it("22. serialises concurrent runs so items are not created twice", async () => {
    const { deps, gh } = mk(FIVE);
    await Promise.all([sync(deps), sync(deps)]);
    expect(gh.count("create")).toBe(5);
  });

  it("23. dry run on a file with conflicts and appends writes nothing", async () => {
    const { h, deps, gh } = mk(THREE);
    await sync(deps);
    gh.seed({ title: "new on gh" });
    gh.issues.get(2)!.state = "CLOSED";
    const writes = gh.writes.length;
    const file = h.file;
    const saves = h.stateSaves;
    const dry = await sync(deps, true);
    expect(dry.actions).toEqual([
      { type: "md-check", issue: 2, checked: true },
      { type: "md-append", issue: 4, title: "new on gh" },
    ]);
    expect(gh.writes).toHaveLength(writes);
    expect(h.file).toBe(file);
    expect(h.stateSaves).toBe(saves);
  });
});

describe("runSync resumability", () => {
  const ONE = "## A\n\n- [ ] one\n";

  it("24. an issue orphaned by a file change during creation is adopted by the next run, not duplicated", async () => {
    const { h, deps, gh } = mk(ONE);
    const edited = ONE + "\n- [ ] user added this meanwhile\n";
    gh.onCreate = () => {
      h.file = edited;
    };
    const r1 = await sync(deps);
    expect(r1.aborted).toBe(true);
    expect(Object.keys(h.state.pending ?? {})).toEqual(["1"]);
    gh.onCreate = () => {};
    const r2 = await sync(deps);
    expect(r2.aborted).toBe(false);
    expect(r2.actions[0]).toEqual({ type: "link", issue: 1, title: "one" });
    expect(gh.count("create")).toBe(2); // "one" once, the user's line once
    expect([...gh.issues.values()].map((i) => i.title).sort()).toEqual(["one", "user added this meanwhile"]);
    expect(h.file).toContain("- [ ] one <!-- gh:#1 -->");
    expect(h.state.pending).toEqual({});
    expect((await sync(deps)).actions).toEqual([]);
  });

  it("25. dry run reports the adoption and changes nothing", async () => {
    const { h, deps, gh } = mk(ONE);
    gh.onCreate = () => {
      h.file = ONE + "\n";
    };
    await sync(deps);
    const writes = gh.writes.length;
    const stateBefore = JSON.stringify(h.state);
    const fileBefore = h.file;
    const r = await sync(deps, true);
    expect(r.actions).toEqual([{ type: "link", issue: 1, title: "one" }]);
    expect(gh.writes).toHaveLength(writes);
    expect(JSON.stringify(h.state)).toBe(stateBefore);
    expect(h.file).toBe(fileBefore);
  });

  it("26. a pending link whose item was deleted is reported under notInFile and not re-appended", async () => {
    const { h, deps, gh } = mk(ONE);
    gh.onCreate = () => {
      h.file = "# nothing left\n";
    };
    await sync(deps);
    gh.onCreate = () => {};
    const r = await sync(deps);
    expect(r.notInFile).toEqual([1]);
    expect(r.actions).toEqual([]);
    expect(h.file).toBe("# nothing left\n");
    expect(gh.count("create")).toBe(1);
  });

  it("27. an edited item (different title or body) is not adopted into the orphan", async () => {
    const { h, deps, gh } = mk(ONE);
    gh.onCreate = () => {
      h.file = "## A\n\n- [ ] one changed\n";
    };
    await sync(deps);
    gh.onCreate = () => {};
    const r = await sync(deps);
    expect(r.actions.map((a) => a.type)).toEqual(["create"]);
    expect(r.notInFile).toEqual([1]);
  });

  it("28. state is saved with the pending link before the marker is written", async () => {
    const { h, deps } = mk(ONE);
    const order: string[] = [];
    const write = deps.writeFile;
    deps.writeFile = (t) => {
      order.push(`write:${Object.keys(h.state.pending ?? {}).join()}`);
      write(t);
    };
    await sync(deps);
    expect(order).toEqual(["write:1"]);
    expect(h.state.pending).toEqual({});
  });

  it("29. a checked item whose issue is open and has no state record is closed, and the box stays ticked", async () => {
    const { h, deps, gh } = mk("## A\n\n- [x] done <!-- gh:#1 -->\n- [ ] other <!-- gh:#2 -->\n");
    gh.seed({ title: "done" });
    gh.seed({ title: "other" });
    h.state.items["2"] = { checked: false, closed: false, bodyHash: crypto.createHash("sha256").update("other").digest("hex") };
    const r = await sync(deps);
    expect(r.actions).toEqual([{ type: "close", issue: 1 }]);
    expect(gh.issues.get(1)!.state).toBe("CLOSED");
    expect(h.file).toBe("## A\n\n- [x] done <!-- gh:#1 -->\n- [ ] other <!-- gh:#2 -->\n");
    expect(h.fileWrites).toBe(0);
    expect((await sync(deps)).actions).toEqual([]);
  });

  it("30. an unchecked item whose issue is closed and has no state record still follows GitHub", async () => {
    const { h, deps, gh } = mk("## A\n\n- [ ] done <!-- gh:#1 -->\n- [ ] other <!-- gh:#2 -->\n");
    gh.seed({ title: "done", state: "CLOSED" });
    gh.seed({ title: "other" });
    h.state.items["2"] = { checked: false, closed: false, bodyHash: crypto.createHash("sha256").update("other").digest("hex") };
    const r = await sync(deps);
    expect(r.actions).toEqual([{ type: "md-check", issue: 1, checked: true }]);
    expect(h.file).toBe("## A\n\n- [x] done <!-- gh:#1 -->\n- [ ] other <!-- gh:#2 -->\n");
    expect(gh.count("reopen")).toBe(0);
  });
});

describe("fileSyncDeps", () => {
  it("reads, atomically rewrites the kanban file and persists state under stateDir", async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "dash-ks-"));
    try {
      const kanbanPath = path.join(dir, "k.md");
      const stateDir = path.join(dir, "state");
      fs.writeFileSync(kanbanPath, "## A\n\n- [ ] one\n", { mode: 0o640 });
      const gh = new FakeGh();
      const deps = fileSyncDeps({ kanbanPath, stateDir } as Config, gh);
      expect(deps.loadState()).toEqual({ labelsEnsured: false, items: {} });
      await runSync(deps, { dryRun: false });
      expect(fs.readFileSync(kanbanPath, "utf8")).toBe("## A\n\n- [ ] one <!-- gh:#1 -->\n");
      expect(fs.statSync(kanbanPath).mode & 0o777).toBe(0o640);
      expect(fs.readdirSync(dir).filter((f) => f.endsWith(".tmp"))).toEqual([]);
      expect(fileSyncDeps({ kanbanPath, stateDir } as Config, gh).loadState().items["1"]).toBeDefined();
      expect(fileSyncDeps({ kanbanPath: path.join(dir, "none.md"), stateDir } as Config, gh).readFile()).toBeNull();
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });
});
