import fs from "node:fs/promises";
import path from "node:path";
import type { AgentEvent, ConversationReadOnly } from "../../shared/types";
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

/** The first message, whitespace collapsed and cut to a title. */
export function autoTitle(text: string): string {
  const flat = text.replace(/\s+/g, " ").trim();
  return flat.length > TITLE_MAX ? `${flat.slice(0, TITLE_MAX)}…` : flat;
}

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

const isEntry = (v: unknown): v is ConvEntry => {
  if (typeof v !== "object" || v === null) return false;
  const e = v as Record<string, unknown>;
  return (
    typeof e.id === "string" && typeof e.createdAt === "string" && typeof e.updatedAt === "string" &&
    typeof e.turns === "number" && typeof e.archived === "boolean" && typeof e.titleManual === "boolean" &&
    (e.title === null || typeof e.title === "string") && (e.sessionId === null || typeof e.sessionId === "string") &&
    (e.costUsd === undefined || e.costUsd === null || typeof e.costUsd === "number") &&
    (e.readOnly === undefined || e.readOnly === null || READ_ONLY_VALUES.includes(e.readOnly as ConversationReadOnly)) &&
    (e.seededFrom === undefined || e.seededFrom === null || typeof e.seededFrom === "string")
  );
};

function parseRegistry(text: string): Registry | null {
  try {
    const v = JSON.parse(text) as Partial<Registry> | null;
    if (!v || v.v !== 1 || typeof v.active !== "string" || typeof v.convs !== "object" || v.convs === null) return null;
    const convs: Record<string, ConvEntry> = {};
    for (const [id, e] of Object.entries(v.convs)) {
      if (CONV_ID.test(id) && isEntry(e) && e.id === id) convs[id] = { ...e, readOnly: e.readOnly ?? null, seededFrom: e.seededFrom ?? null, costUsd: e.costUsd ?? null };
    }
    if (Object.keys(convs).length === 0) return null;
    // An unsafe or dangling pointer must not cost the user their conversations: keep them, with no active one.
    const active = CONV_ID.test(v.active) && v.active in convs ? v.active : "";
    return { v: 1, active, convs };
  } catch {
    return null;
  }
}

/** The registry from `index.json`, or from its backup when that is unreadable; null when neither exists. */
export async function loadRegistry(dir: string): Promise<Registry | null> {
  for (const name of ["index.json", "index.json.bak"]) {
    try {
      const reg = parseRegistry(await fs.readFile(path.join(dir, name), "utf8"));
      if (reg) return reg;
    } catch {
      /* missing: try the next copy */
    }
  }
  return null;
}

/** Atomic write; the previous good copy is kept as `index.json.bak` (a corrupt one never replaces it). */
export async function saveRegistry(dir: string, reg: Registry): Promise<void> {
  const file = path.join(dir, "index.json");
  let current: string | null = null;
  try {
    current = await fs.readFile(file, "utf8");
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code !== "ENOENT") throw err;
  }
  if (current !== null && parseRegistry(current)) {
    const bakTmp = `${file}.bak.${process.pid}.tmp`;
    await fs.writeFile(bakTmp, current);
    await fs.rename(bakTmp, `${file}.bak`);
  }
  const tmp = `${file}.${process.pid}.tmp`;
  await fs.writeFile(tmp, JSON.stringify(reg, null, 2));
  await fs.rename(tmp, file);
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
      title: stats.firstUser ? autoTitle(stats.firstUser) : null,
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
      title: stats.firstUser ? autoTitle(stats.firstUser) : null,
      updatedAt: stats.lastAt ?? nowIso,
      turns: stats.turns,
      costUsd: stats.costUsd,
      archived: true,
      readOnly: "legacy",
    };
  }

  if (active === null) {
    active = newConvId(Date.parse(nowIso));
    convs[active] = blankEntry(active, nowIso);
  }
  const reg: Registry = { v: 1, active, convs };
  await saveRegistry(dir, reg);
  await fs.rename(oldMeta, `${oldMeta}.migrated`).catch(() => undefined);
  return reg;
}
