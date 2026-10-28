import fs from "node:fs/promises";
import path from "node:path";
import type { AgentEvent, ConversationReadOnly } from "../../shared/types";
import { sweepTemp, writeAtomic } from "../lib/atomicFile";
import { readJournalFile } from "./journal";

/** Conversation ids become directory names, so only this shape is accepted from outside. */
const CONV_ID_MAX_LENGTH = 48;
export const CONV_ID = new RegExp(`^c-[a-z0-9-]{1,${CONV_ID_MAX_LENGTH}}$`);
export const TITLE_MAX = 60;
export const TITLE_EDIT_MAX = 120;
/** The conversation that was live before registries existed keeps this fixed id so an interrupted import can be re-run. */
export const ACTIVE_LEGACY_ID = "c-migrated-active";
const RAND_DIGITS = 4;
const RAND_RANGE = 36 ** RAND_DIGITS;

export interface ConvEntry {
  id: string;
  title: string | null;
  /** A title the owner typed is never replaced by the automatic one. */
  titleManual: boolean;
  createdAt: string;
  updatedAt: string;
  turns: number;
  costUsd: number | null;
  /** The SDK session to resume; null once it is known to be gone. */
  sessionId: string | null;
  archived: boolean;
  readOnly: ConversationReadOnly | null;
  seededFrom: string | null;
}

export interface Registry {
  v: 1;
  active: string;
  convs: Record<string, ConvEntry>;
}

export interface JournalStats {
  firstUser: string | null;
  turns: number;
  costUsd: number | null;
  firstAt: string | null;
  lastAt: string | null;
}

export const newConvId = (nowMs: number, rand: () => number = Math.random): string =>
  `c-${nowMs.toString(36)}${Math.floor(rand() * RAND_RANGE).toString(36).padStart(RAND_DIGITS, "0")}`;

export const blankEntry = (id: string, at: string, seededFrom: string | null = null): ConvEntry => ({
  id, title: null, titleManual: false, createdAt: at, updatedAt: at, turns: 0, costUsd: null, sessionId: null, archived: false, readOnly: null, seededFrom,
});

export const convDir = (dir: string, id: string): string => path.join(dir, "conv", id);
export const journalPath = (dir: string, id: string): string => path.join(convDir(dir, id), "journal.jsonl");

/** One line: whitespace and control characters become single spaces. Used for every title, typed or automatic. */
// eslint-disable-next-line no-control-regex
export const cleanTitle = (text: string): string => text.replace(/[\s\u0000-\u001f\u007f]+/g, " ").trim();

/** The first message, cleaned and cut to a title; "" when nothing printable is left (callers treat that as no title). */
export function autoTitle(text: string): string {
  const chars = [...cleanTitle(text)]; // by code point: a cut must not split an emoji
  return chars.length > TITLE_MAX ? `${chars.slice(0, TITLE_MAX).join("")}…` : chars.join("");
}

/** Nothing worth keeping: never messaged, untitled, not seeded from another conversation. */
export const isEmptyConv = (e: ConvEntry): boolean => e.turns === 0 && e.sessionId === null && e.title === null && e.seededFrom === null && e.readOnly === null;

/** The newest conversation that can still be continued. */
export const pickContinuable = (convs: Record<string, ConvEntry>): ConvEntry | undefined =>
  Object.values(convs).filter((c) => !c.archived && !c.readOnly).sort((a, b) => b.updatedAt.localeCompare(a.updatedAt))[0];

export function statsOf(events: AgentEvent[]): JournalStats {
  let firstUser: string | null = null;
  let turns = 0;
  let costUsd: number | null = null;
  for (const e of events) {
    if (e.kind === "user" && firstUser === null) firstUser = e.text;
    if (e.kind === "result") {
      turns += 1;
      if (typeof e.costUsd === "number") costUsd = Math.max(costUsd ?? 0, e.costUsd);
    }
  }
  return { firstUser, turns, costUsd, firstAt: events[0]?.at ?? null, lastAt: events.length > 0 ? events[events.length - 1].at : null };
}

const READ_ONLY_VALUES: readonly ConversationReadOnly[] = ["expired", "legacy"];

/**
 * An entry from disk, with odd field values coerced instead of the entry dropped: a conversation lost from the
 * index is lost for good, while an unknown `readOnly` (a newer build's) is safely treated as "legacy".
 * Only an entry with no usable structure (id and dates) is refused.
 */
const toEntry = (id: string, v: unknown): ConvEntry | null => {
  if (typeof v !== "object" || v === null) return null;
  const e = v as Record<string, unknown>;
  if (e.id !== id || typeof e.createdAt !== "string" || typeof e.updatedAt !== "string") return null;
  const text = (x: unknown): string | null => (typeof x === "string" ? x : null);
  const count = (x: unknown): number | null => (typeof x === "number" && Number.isFinite(x) ? x : null);
  const readOnly = e.readOnly == null ? null : READ_ONLY_VALUES.includes(e.readOnly as ConversationReadOnly) ? (e.readOnly as ConversationReadOnly) : "legacy";
  return {
    id, title: text(e.title), titleManual: e.titleManual === true, createdAt: e.createdAt, updatedAt: e.updatedAt, turns: count(e.turns) ?? 0,
    costUsd: count(e.costUsd), sessionId: text(e.sessionId), archived: e.archived === true, readOnly, seededFrom: text(e.seededFrom),
  };
};

function parseRegistry(text: string): Registry | null {
  try {
    const v = JSON.parse(text) as Partial<Registry> | null;
    if (!v || v.v !== 1 || typeof v.active !== "string" || typeof v.convs !== "object" || v.convs === null) return null;
    const convs: Record<string, ConvEntry> = {};
    for (const [id, e] of Object.entries(v.convs)) {
      const entry = CONV_ID.test(id) ? toEntry(id, e) : null;
      if (entry) convs[id] = entry;
    }
    if (Object.keys(convs).length === 0) return null;
    // An unsafe or dangling pointer must not cost the user their conversations: keep them, with no active one.
    const active = CONV_ID.test(v.active) && Object.hasOwn(convs, v.active) ? v.active : "";
    return { v: 1, active, convs };
  } catch {
    return null;
  }
}

/** The index, the copy from before the last save, and the copy from before today's first save. */
const INDEX_COPIES = ["index.json", "index.json.bak", "index.json.daily"];

/** The registry from `index.json`, or from the newest backup that is readable; null when none is. */
export async function loadRegistry(dir: string): Promise<Registry | null> {
  for (const name of INDEX_COPIES) {
    try {
      const reg = parseRegistry(await fs.readFile(path.join(dir, name), "utf8"));
      if (reg) return reg;
    } catch {
      /* missing: try the next copy */
    }
  }
  return null;
}

/**
 * What one writer last put in `index.json`, with the file's stamp, to tell whether anyone else changed it.
 * It belongs to the writer (the chief keeps one), not to the module: two writers on one directory must not share it.
 */
export interface WriteMemo {
  last?: { text: string; mtimeMs: number; size: number };
}

const sameDay = (a: number, b: number): boolean => new Date(a).toISOString().slice(0, 10) === new Date(b).toISOString().slice(0, 10);

/** Removes temp files that a killed process left in the state directory and in each conversation folder. */
export async function sweepTempFiles(dir: string): Promise<void> {
  await sweepTemp(dir);
  for (const id of await fs.readdir(path.join(dir, "conv")).catch(() => [] as string[])) {
    if (CONV_ID.test(id)) await sweepTemp(convDir(dir, id));
  }
}

const entryFile = (dir: string, id: string): string => path.join(convDir(dir, id), "entry.json");

/**
 * Each conversation also keeps its own `entry.json` beside its journal, so the index can be rebuilt without losing
 * session ids: the index is a fast copy of these files, not the only place the information lives.
 */
async function writeEntry(dir: string, e: ConvEntry): Promise<void> {
  await fs.mkdir(convDir(dir, e.id), { recursive: true });
  await writeAtomic(entryFile(dir, e.id), JSON.stringify(e, null, 2));
}

async function readEntry(dir: string, id: string): Promise<ConvEntry | null> {
  try {
    return toEntry(id, JSON.parse(await fs.readFile(entryFile(dir, id), "utf8")));
  } catch {
    return null;
  }
}

/**
 * Takes the on-disk `entry.json` of every conversation folder over what `convs` says: an entry the index lost is
 * brought back, and one saved just before a crash (entries are written before the index) wins over a stale index.
 */
export async function adoptEntries(dir: string, convs: Record<string, ConvEntry>): Promise<void> {
  for (const id of await fs.readdir(path.join(dir, "conv")).catch(() => [] as string[])) {
    if (!CONV_ID.test(id)) continue;
    const saved = await readEntry(dir, id);
    if (saved) convs[id] = saved;
  }
}

/**
 * Atomic write. The previous good copy is kept as `index.json.bak` and the first good copy of each day as
 * `index.json.daily` (a corrupt one never replaces either). `entryIds` are the conversations that changed: their
 * `entry.json` is written first, so a crash between the two writes leaves the newer truth on disk.
 */
export async function saveRegistry(dir: string, reg: Registry, entryIds: readonly string[] = [], memo: WriteMemo = {}): Promise<void> {
  const file = path.join(dir, "index.json");
  for (const id of entryIds) if (Object.hasOwn(reg.convs, id)) await writeEntry(dir, reg.convs[id]);
  const text = JSON.stringify(reg, null, 2);
  let current: string | null = null;
  let valid = false;
  const st = await fs.stat(file).catch(() => null);
  const last = memo.last;
  if (st && last && last.mtimeMs === st.mtimeMs && last.size === st.size) {
    // The file is still exactly what this process wrote (and validated), so it is not read and parsed again.
    current = last.text;
    valid = true;
  } else if (st) {
    current = await fs.readFile(file, "utf8");
    valid = parseRegistry(current) !== null;
  }
  if (valid && current === text) return; // nothing to save
  if (current !== null && valid) {
    await writeAtomic(`${file}.bak`, current);
    const daily = await fs.stat(`${file}.daily`).catch(() => null);
    if (!daily || !sameDay(daily.mtimeMs, Date.now())) await writeAtomic(`${file}.daily`, current);
  }
  await writeAtomic(file, text);
  const after = await fs.stat(file);
  memo.last = { text, mtimeMs: after.mtimeMs, size: after.size };
}

const exists = (p: string): Promise<boolean> => fs.access(p).then(() => true, () => false);
const moveTo = async (from: string, to: string): Promise<void> => {
  await fs.mkdir(path.dirname(to), { recursive: true });
  await fs.rename(from, to);
};

/**
 * Builds a registry from what is on disk: the pre-registry `journal.jsonl` + `meta.json` become the active
 * conversation (session id kept), each `journal-<stamp>.jsonl` becomes a read-only archived one, and any
 * other conversation folder is recovered the same way. Safe to run again after an interruption.
 */
export async function rebuildRegistry(dir: string, nowIso: string): Promise<Registry> {
  await fs.mkdir(dir, { recursive: true });
  const convs: Record<string, ConvEntry> = {};
  const oldJournal = path.join(dir, "journal.jsonl");
  const oldMeta = path.join(dir, "meta.json");
  const activeFile = journalPath(dir, ACTIVE_LEGACY_ID);

  if (await exists(oldJournal)) await moveTo(oldJournal, activeFile);
  let saved: { turns?: unknown; costUsd?: unknown; sessionId?: unknown } = {};
  try {
    saved = JSON.parse(await fs.readFile(oldMeta, "utf8"));
  } catch {
    /* no meta */
  }
  let active: string | null = null;
  if ((await exists(activeFile)) || typeof saved.sessionId === "string") {
    const stats = statsOf(await readJournalFile(activeFile));
    convs[ACTIVE_LEGACY_ID] = {
      ...blankEntry(ACTIVE_LEGACY_ID, stats.firstAt ?? nowIso),
      title: autoTitle(stats.firstUser ?? "") || null,
      updatedAt: stats.lastAt ?? nowIso,
      turns: typeof saved.turns === "number" ? saved.turns : stats.turns,
      costUsd: typeof saved.costUsd === "number" ? saved.costUsd : stats.costUsd,
      sessionId: typeof saved.sessionId === "string" ? saved.sessionId : null,
    };
    active = ACTIVE_LEGACY_ID;
  }

  for (const name of await fs.readdir(dir).catch(() => [] as string[])) {
    const m = /^journal-(.+)\.jsonl$/.exec(name);
    const id = m ? `c-${m[1].toLowerCase()}` : "";
    if (CONV_ID.test(id)) await moveTo(path.join(dir, name), journalPath(dir, id));
  }
  for (const id of await fs.readdir(path.join(dir, "conv")).catch(() => [] as string[])) {
    if (convs[id] || !CONV_ID.test(id)) continue;
    const stats = statsOf(await readJournalFile(journalPath(dir, id)));
    convs[id] = {
      ...blankEntry(id, stats.firstAt ?? nowIso),
      title: autoTitle(stats.firstUser ?? "") || null,
      updatedAt: stats.lastAt ?? nowIso,
      turns: stats.turns,
      costUsd: stats.costUsd,
      archived: true,
      readOnly: "legacy",
    };
  }

  // Conversations that saved their own entry come back whole (session id included), not as read-only archives.
  await adoptEntries(dir, convs);
  if (active === null || convs[active]?.readOnly) active = pickContinuable(convs)?.id ?? null;
  if (active === null) {
    active = newConvId(Date.parse(nowIso));
    convs[active] = blankEntry(active, nowIso);
  }
  const reg: Registry = { v: 1, active, convs };
  await saveRegistry(dir, reg, Object.keys(convs));
  await fs.rename(oldMeta, `${oldMeta}.migrated`).catch(() => undefined);
  return reg;
}
