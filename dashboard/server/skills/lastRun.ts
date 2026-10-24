import fs from "node:fs";
import path from "node:path";

const SKILL_TOOL_RE = /"name":\s*"Skill"/;
const COMMAND_MARK = "<command-name>/";
/** A slash command message starts with its tags (Claude Code writes `<command-message>` first); text that merely quotes them does not. */
const COMMAND_RE = /^\s*(?:<command-message>[\s\S]*?<\/command-message>\s*)?<command-name>\/([A-Za-z0-9:_-]+)<\/command-name>/;
const CHUNK_BYTES = 4 * 1024 * 1024;
const NEWLINE = 0x0a;
/** Skill calls still waiting for their result; an unanswered call is dropped once this many are newer. */
const PENDING_MAX = 50;
/** Claude Code names a worktree's transcript folder `<repo folder>--claude-worktrees-<name>`. */
const WORKTREE_INFIX = "--claude-worktrees-";

interface Block {
  type?: string;
  name?: string;
  id?: string;
  tool_use_id?: string;
  is_error?: boolean;
  text?: string;
  input?: { skill?: unknown };
}
interface Entry {
  timestamp?: unknown;
  message?: { role?: unknown; content?: unknown };
}

/** What has been learned from one transcript so far. Transcripts only grow at the end, so `offset` marks where to resume. */
interface FileState {
  offset: number;
  runs: Map<string, string>;
  pending: Map<string, { name: string; at: string }>;
}

const files = new Map<string, FileState>();
const inflight = new Map<string, Promise<Map<string, string>>>();
/** Scans run one at a time: two scans over different folder lists could otherwise read the same transcript from the same offset. */
let queue: Promise<unknown> = Promise.resolve();

function newer(runs: Map<string, string>, name: string, at: string): void {
  const prev = runs.get(name);
  if (!prev || Date.parse(at) > Date.parse(prev)) runs.set(name, at);
}

function firstText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const block = (content as Block[]).find((b) => b?.type === "text" && typeof b.text === "string");
  return block?.text ?? "";
}

/** Reads one transcript line; only skill names and entry times are kept. */
function answersPending(line: string, st: FileState): boolean {
  if (st.pending.size === 0 || !line.includes('"tool_result"')) return false;
  for (const id of st.pending.keys()) if (line.includes(id)) return true;
  return false;
}

function processLine(line: string, st: FileState): void {
  const wantsResult = answersPending(line, st);
  if (!wantsResult && !SKILL_TOOL_RE.test(line) && !line.includes(COMMAND_MARK)) return;
  let entry: Entry;
  try {
    entry = JSON.parse(line);
  } catch {
    return;
  }
  const content = entry.message?.content;
  const blocks = Array.isArray(content) ? (content as Block[]) : [];
  for (const b of blocks) {
    if (b?.type === "tool_result" && typeof b.tool_use_id === "string") {
      const call = st.pending.get(b.tool_use_id);
      if (!call) continue;
      st.pending.delete(b.tool_use_id);
      if (b.is_error !== true) newer(st.runs, call.name, call.at); // a call that errored did not run the skill
    }
  }
  const at = entry.timestamp;
  if (typeof at !== "string" || Number.isNaN(Date.parse(at))) return;
  for (const b of blocks) {
    if (b?.type === "tool_use" && b.name === "Skill" && typeof b.id === "string" && typeof b.input?.skill === "string") {
      st.pending.set(b.id, { name: b.input.skill, at });
      if (st.pending.size > PENDING_MAX) st.pending.delete(st.pending.keys().next().value as string);
    }
  }
  if (entry.message?.role === "user") {
    const m = COMMAND_RE.exec(firstText(content));
    if (m) newer(st.runs, m[1], at);
  }
}

/** Reads `st.offset` to `size`, in chunks cut at line ends, and moves the offset past the last complete line. */
async function readNew(file: string, st: FileState, size: number): Promise<void> {
  const fh = await fs.promises.open(file, "r");
  try {
    const buf = Buffer.allocUnsafe(Math.min(CHUNK_BYTES, size - st.offset));
    let pos = st.offset;
    let carry = Buffer.alloc(0);
    while (pos < size) {
      const { bytesRead } = await fh.read(buf, 0, Math.min(buf.length, size - pos), pos);
      if (bytesRead === 0) break;
      pos += bytesRead;
      const chunk = Buffer.concat([carry, buf.subarray(0, bytesRead)]);
      const nl = chunk.lastIndexOf(NEWLINE);
      if (nl < 0) {
        carry = chunk;
        continue;
      }
      for (const line of chunk.subarray(0, nl).toString("utf8").split("\n")) if (line) processLine(line, st);
      carry = chunk.subarray(nl + 1);
    }
    st.offset = pos - carry.length;
  } finally {
    await fh.close();
  }
}

async function scan(dirs: string[]): Promise<Map<string, string>> {
  const seen = new Set<string>();
  const merged = new Map<string, string>();
  for (const dir of dirs) {
    let names: string[];
    try {
      names = (await fs.promises.readdir(dir)).filter((n) => n.endsWith(".jsonl"));
    } catch {
      continue;
    }
    for (const n of names) {
      const file = path.join(dir, n);
      seen.add(file);
      try {
        const { size } = await fs.promises.stat(file);
        let st = files.get(file);
        if (!st || size < st.offset) {
          st = { offset: 0, runs: new Map(), pending: new Map() }; // new file, or one that was rewritten shorter
          files.set(file, st);
        }
        if (size > st.offset) await readNew(file, st, size);
        for (const [name, at] of st.runs) newer(merged, name, at);
      } catch {
        // an unreadable transcript is skipped, not fatal
      }
    }
  }
  for (const f of files.keys()) if (dirs.includes(path.dirname(f)) && !seen.has(f)) files.delete(f);
  return merged;
}

/**
 * Newest time each skill was invoked in the Claude Code transcripts under `dirs`: a Skill tool call that got a
 * successful result, or a /slash command typed by the user. Returns skill name to ISO time. Only the part of a
 * transcript added since the last scan is read, and requests that overlap share one scan.
 */
export function scanLastRuns(dirs: string | string[]): Promise<Map<string, string>> {
  const list = [...new Set(Array.isArray(dirs) ? dirs : [dirs])];
  const key = list.join("\0");
  const running = inflight.get(key);
  if (running) return running;
  const p = queue.then(() => scan(list)).finally(() => inflight.delete(key));
  queue = p.catch(() => {});
  inflight.set(key, p);
  return p;
}

/** `mainDir` plus the transcript folders of its worktrees, so runs from a feature worktree count too. */
export async function withWorktreeDirs(mainDir: string): Promise<string[]> {
  const parent = path.dirname(mainDir);
  const prefix = path.basename(mainDir) + WORKTREE_INFIX;
  try {
    const names = (await fs.promises.readdir(parent)).filter((n) => n.startsWith(prefix)).sort();
    return [mainDir, ...names.map((n) => path.join(parent, n))];
  } catch {
    return [mainDir];
  }
}
