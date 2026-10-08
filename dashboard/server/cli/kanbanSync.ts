import { pathToFileURL } from "node:url";
import { loadConfig } from "../config";
import { createGh } from "../lib/gh";
import { execRunner } from "../runner";
import { fileSyncDeps, isImportCompleted, removeStaleTempFiles, runSync, SyncLockedError, type SyncAction, type SyncDeps, type SyncReport } from "../lib/kanbanSync";

export const USAGE = [
  "Usage: npm run kanban:sync -- [--dry-run] [--yes] [--rebuild-state]",
  "  --dry-run        print what a sync would do; makes no gh write and writes no file",
  "  --yes            confirm the first import (required until a run has completed the import)",
  "  --rebuild-state  rebuild the sync state from the file and GitHub when it was lost; flips and closes nothing",
  "  --help           show this text",
].join("\n");

export interface CliIo {
  sync: SyncDeps;
  /** True once a real run has completed the first import (state.importCompletedAt). */
  importCompleted(): boolean;
  out(line: string): void;
  err(line: string): void;
}

interface Tense {
  create: string;
  createClose: string;
  link: string;
  update: string;
  append: string;
}
const PLANNED: Tense = { create: "to create", createClose: "to create and close", link: "to link", update: "to update", append: "to append" };
const DONE: Tense = { create: "created", createClose: "created and closed", link: "linked", update: "updated", append: "appended" };

function describeAction(a: SyncAction): string {
  switch (a.type) {
    case "create":
      return `${a.closed ? "create and close" : "create"}: ${a.title}`;
    case "link":
      return `link #${a.issue} to its item in the file: ${a.title}`;
    case "close":
      return `close #${a.issue}`;
    case "reopen":
      return `reopen #${a.issue}`;
    case "update-body":
      return `update body of #${a.issue}`;
    case "md-check":
      return `${a.checked ? "tick" : "untick"} #${a.issue} in the file`;
    case "md-append":
      return `append to Inbox: #${a.issue} ${a.title}`;
  }
}

function printReport(r: SyncReport, tense: Tense, out: (l: string) => void): void {
  const count = (pred: (a: SyncAction) => boolean) => r.actions.filter(pred).length;
  out(`${tense.create}: ${count((a) => a.type === "create" && !a.closed)}`);
  out(`${tense.createClose}: ${count((a) => a.type === "create" && a.closed)}`);
  out(`${tense.link}: ${count((a) => a.type === "link")}`);
  out(`${tense.update}: ${count((a) => ["close", "reopen", "update-body", "md-check"].includes(a.type))}`);
  out(`${tense.append}: ${count((a) => a.type === "md-append")}`);
  out(`conflicts: ${r.conflicts.length}`);
  out(`skipped: ${r.skipped.length}`);
  out(`not in file: ${r.notInFile.length}`);
  for (const a of r.actions) out(`  ${describeAction(a)}`);
  for (const c of r.conflicts) out(`  conflict: ${c}`);
  for (const s of r.skipped) out(`  skipped "${s.title}" (${s.section}): ${s.reason}`);
  for (const n of r.notInFile) out(`  not in file: #${n}`);
}

const oneLine = (s: string): string => s.replace(/\s+/g, " ").trim();

export async function main(argv: string[], io: CliIo): Promise<number> {
  const known = new Set(["--dry-run", "--yes", "--rebuild-state", "--help"]);
  const bad = argv.filter((a) => !known.has(a));
  if (bad.length > 0) {
    io.err(`unknown argument: ${bad.join(" ")}`);
    io.err(USAGE);
    return 2;
  }
  if (argv.includes("--help")) {
    io.out(USAGE);
    return 0;
  }
  const dryRun = argv.includes("--dry-run");
  const yes = argv.includes("--yes");
  const rebuildState = argv.includes("--rebuild-state");
  try {
    const firstImport = !io.importCompleted();
    if (dryRun || (firstImport && !yes)) {
      const report = await runSync(io.sync, { dryRun: true, rebuildState });
      if (dryRun) io.out("Dry run: nothing was changed.");
      printReport(report, PLANNED, io.out);
      if (dryRun) return 0;
      io.err("This is the first import (or an interrupted one): it will create GitHub issues and add markers to the kanban file.");
      io.err("Review the plan above, then run: npm run kanban:sync -- --yes");
      return 1;
    }
    const report = await runSync(io.sync, { dryRun: false, rebuildState });
    printReport(report, DONE, io.out);
    if (report.aborted) {
      io.err(`sync aborted: ${oneLine(report.conflicts[report.conflicts.length - 1] ?? "the kanban file changed on disk")}; run it again`);
      return 1;
    }
    if (!io.importCompleted()) {
      io.err(
        "the import is not complete: some items could not be created or linked (see the conflicts and skipped items above), " +
          "so the server's background sync stays off; fix them and re-run: npm run kanban:sync -- --yes",
      );
      return 1;
    }
    return 0;
  } catch (err) {
    if (err instanceof SyncLockedError) {
      io.err(oneLine(err.message));
      return 1;
    }
    io.err(`kanban sync failed: ${oneLine(err instanceof Error ? err.message : String(err))}`);
    return 1;
  }
}

async function cli(): Promise<void> {
  const config = loadConfig(process.env);
  const gh = createGh(execRunner, config);
  if (!process.argv.includes("--dry-run") && !process.argv.includes("--help")) removeStaleTempFiles(config);
  const code = await main(process.argv.slice(2), {
    sync: fileSyncDeps(config, gh),
    importCompleted: () => isImportCompleted(config),
    out: (l) => console.log(l),
    err: (l) => console.error(l),
  });
  process.exitCode = code;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  cli().catch((err) => {
    console.error(`kanban sync failed: ${oneLine(err instanceof Error ? err.message : String(err))}`);
    process.exitCode = 1;
  });
}
