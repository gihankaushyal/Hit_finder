import fs from "node:fs";
import path from "node:path";
import readline from "node:readline";

const SKILL_TOOL_MARK = '"name":"Skill"';
const COMMAND_MARK = "<command-name>/";
const COMMAND_RE = /<command-name>\/([A-Za-z0-9:_-]+)<\/command-name>/g;

interface FileRuns {
  mtimeMs: number;
  size: number;
  runs: Map<string, string>;
}

/** Per-file results, so only transcripts that changed since the last scan are read again. */
const cache = new Map<string, FileRuns>();

function newer(runs: Map<string, string>, name: string, at: string): void {
  const prev = runs.get(name);
  if (!prev || Date.parse(at) > Date.parse(prev)) runs.set(name, at);
}

function textOf(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content.map((b) => (b && typeof b === "object" && typeof (b as { text?: unknown }).text === "string" ? (b as { text: string }).text : "")).join("\n");
}

/** Reads one transcript line; only the skill name and the entry time leave this function. */
function collect(line: string, runs: Map<string, string>): void {
  let entry: { timestamp?: unknown; message?: { role?: unknown; content?: unknown } };
  try {
    entry = JSON.parse(line);
  } catch {
    return;
  }
  const at = entry.timestamp;
  if (typeof at !== "string" || Number.isNaN(Date.parse(at))) return;
  const content = entry.message?.content;
  if (Array.isArray(content)) {
    for (const block of content as Array<{ type?: string; name?: string; input?: { skill?: unknown } }>) {
      if (block?.type === "tool_use" && block.name === "Skill" && typeof block.input?.skill === "string") newer(runs, block.input.skill, at);
    }
  }
  const text = textOf(content);
  if (entry.message?.role === "user" && text.includes(COMMAND_MARK)) for (const m of text.matchAll(COMMAND_RE)) newer(runs, m[1], at);
}

async function scanFile(file: string): Promise<Map<string, string>> {
  const runs = new Map<string, string>();
  const lines = readline.createInterface({ input: fs.createReadStream(file, { encoding: "utf8" }), crlfDelay: Infinity });
  for await (const line of lines) {
    if (line.includes(SKILL_TOOL_MARK) || line.includes(COMMAND_MARK)) collect(line, runs);
  }
  return runs;
}

/**
 * Newest time each skill was invoked in the Claude Code transcripts under `dir` (tool calls and /slash commands).
 * Returns skill name to ISO time; a missing directory gives an empty map.
 */
export async function scanLastRuns(dir: string): Promise<Map<string, string>> {
  let names: string[];
  try {
    names = (await fs.promises.readdir(dir)).filter((n) => n.endsWith(".jsonl"));
  } catch {
    return new Map();
  }
  const seen = new Set<string>();
  const merged = new Map<string, string>();
  for (const n of names) {
    const file = path.join(dir, n);
    seen.add(file);
    let st: fs.Stats;
    try {
      st = await fs.promises.stat(file);
    } catch {
      continue;
    }
    let hit = cache.get(file);
    if (!hit || hit.mtimeMs !== st.mtimeMs || hit.size !== st.size) {
      try {
        hit = { mtimeMs: st.mtimeMs, size: st.size, runs: await scanFile(file) };
      } catch {
        continue;
      }
      cache.set(file, hit);
    }
    for (const [name, at] of hit.runs) newer(merged, name, at);
  }
  for (const f of cache.keys()) if (path.dirname(f) === dir && !seen.has(f)) cache.delete(f);
  return merged;
}
