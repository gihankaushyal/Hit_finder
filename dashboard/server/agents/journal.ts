import fs from "node:fs/promises";
import path from "node:path";
import type { AgentEvent, AgentEventBody } from "../../shared/types";

export const JOURNAL_MAX_BYTES = 8 * 1024 * 1024;
export const JOURNAL_TRUNCATED_TEXT = "Log size limit reached; later entries were not recorded.";

export interface Journal {
  append(body: AgentEventBody): Promise<AgentEvent>;
  /** Entries with seq greater than `after`, oldest first, at most `limit`. */
  read(after: number, limit: number): Promise<AgentEvent[]>;
  lastSeq(): number;
}

function parseLine(line: string): AgentEvent | null {
  if (!line.trim()) return null;
  try {
    const v = JSON.parse(line) as AgentEvent;
    return v && typeof v === "object" && Number.isInteger(v.seq) && typeof v.kind === "string" ? v : null;
  } catch {
    return null;
  }
}

async function readEntries(file: string): Promise<AgentEvent[]> {
  let text: string;
  try {
    text = await fs.readFile(file, "utf8");
  } catch (e) {
    if ((e as NodeJS.ErrnoException).code === "ENOENT") return [];
    throw e;
  }
  const out: AgentEvent[] = [];
  for (const line of text.split("\n")) {
    const e = parseLine(line);
    if (e) out.push(e);
  }
  return out;
}

/** The entries with seq greater than `after` from a list sorted by seq, at most `limit` (binary search). */
export function eventsAfter(entries: readonly AgentEvent[], after: number, limit: number): AgentEvent[] {
  let lo = 0;
  let hi = entries.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (entries[mid].seq > after) hi = mid;
    else lo = mid + 1;
  }
  return entries.slice(lo, lo + Math.max(0, limit));
}

/**
 * Reads a journal without opening it for writing: a torn last line is skipped and the file is never
 * modified (unlike `openJournal`, which repairs it). For parked conversations.
 */
export async function readJournalFile(file: string): Promise<AgentEvent[]> {
  return (await readEntries(file)).sort((a, b) => a.seq - b.seq);
}

/**
 * Append-only JSONL log for one agent. Appends are serialised, so order and sequence numbers
 * match call order. A torn last line (crash mid-write) is ignored on read, and the next append
 * starts on a fresh line.
 */
export async function openJournal(file: string, opts: { maxBytes?: number } = {}): Promise<Journal> {
  const maxBytes = opts.maxBytes ?? JOURNAL_MAX_BYTES;
  await fs.mkdir(path.dirname(file), { recursive: true });
  const existing = await readEntries(file);
  let seq = existing.reduce((m, e) => Math.max(m, e.seq), 0);
  let truncated = existing.some((e) => e.kind === "error" && e.message === JOURNAL_TRUNCATED_TEXT);
  let bytes = 0;
  try {
    const raw = await fs.readFile(file);
    bytes = raw.length;
    if (bytes > 0 && raw[bytes - 1] !== 0x0a) {
      await fs.appendFile(file, "\n");
      bytes += 1;
    }
  } catch (e) {
    if ((e as NodeJS.ErrnoException).code !== "ENOENT") throw e;
  }

  // Entries stay in memory (bounded by maxBytes), so a read never re-parses the file.
  const entries: AgentEvent[] = existing.slice().sort((a, b) => a.seq - b.seq);
  let chain: Promise<unknown> = Promise.resolve();
  const write = async (entry: AgentEvent): Promise<void> => {
    const line = JSON.stringify(entry) + "\n";
    await fs.appendFile(file, line);
    bytes += Buffer.byteLength(line);
    entries.push(entry);
  };

  const append = (body: AgentEventBody): Promise<AgentEvent> => {
    const run = async (): Promise<AgentEvent> => {
      const at = new Date().toISOString();
      const line = JSON.stringify({ ...body, seq: seq + 1, at }) + "\n";
      if (truncated || bytes + Buffer.byteLength(line) > maxBytes) {
        if (!truncated) {
          truncated = true;
          seq += 1;
          const marker: AgentEvent = { kind: "error", message: JOURNAL_TRUNCATED_TEXT, seq, at };
          await write(marker);
          return marker;
        }
        return { kind: "error", message: JOURNAL_TRUNCATED_TEXT, seq, at };
      }
      seq += 1;
      const entry = { ...body, seq, at } as AgentEvent;
      await write(entry);
      return entry;
    };
    const p = chain.then(run, run);
    chain = p.catch(() => undefined);
    return p;
  };

  return {
    append,
    async read(after, limit) {
      await chain;
      return eventsAfter(entries, after, limit);
    },
    lastSeq: () => seq,
  };
}
