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

  let chain: Promise<unknown> = Promise.resolve();
  const write = async (entry: AgentEvent): Promise<void> => {
    const line = JSON.stringify(entry) + "\n";
    await fs.appendFile(file, line);
    bytes += Buffer.byteLength(line);
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
      const all = await readEntries(file);
      return all.filter((e) => e.seq > after).slice(0, Math.max(0, limit));
    },
    lastSeq: () => seq,
  };
}
