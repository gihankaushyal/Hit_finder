import type { RunRow, RunStatus } from "../../../shared/types";

export type SortKey = "name" | "status" | "detector" | "crossAp" | "crossF1" | "inDomainAp" | "epochs" | "updated";
export type SortDir = "asc" | "desc";

/**
 * One entry per column. Being a Record, adding a SortKey without describing it here fails to compile, so the
 * default direction and the spoken wording cannot silently fall out of step with the column list.
 */
const KIND: Record<SortKey, "text" | "status" | "number" | "time"> = {
  name: "text",
  status: "status",
  detector: "text",
  crossAp: "number",
  crossF1: "number",
  inDomainAp: "number",
  epochs: "number",
  updated: "time",
};

export const SORT_KEYS = Object.keys(KIND) as SortKey[];

/** Ascending status order lists the runs that need a human first. */
const STATUS_RANK: Record<RunStatus, number> = { provisional: 0, unknown: 1, "no-results": 2, pretrain: 3, complete: 4 };

// Numeric collation puts -v2 before -v10; base sensitivity ignores case (ePix10k next to EPIX10K).
const collator = new Intl.Collator("en", { numeric: true, sensitivity: "base" });

/** Text columns start A to Z; numbers and times start with the highest or newest. */
export function defaultDirection(key: SortKey): SortDir {
  return KIND[key] === "text" || KIND[key] === "status" ? "asc" : "desc";
}

type Value = number | string | null;

function metricOf(block: Record<string, number | null> | null | undefined, name: string): number | null {
  const v = block?.[name];
  return typeof v === "number" ? v : null;
}

function valueOf(r: RunRow, key: SortKey): Value {
  switch (key) {
    case "name": return r.name;
    case "status": return STATUS_RANK[r.status];
    case "detector": return r.result?.testDetector ?? null;
    case "crossAp": return metricOf(r.result?.cross, "ap");
    case "crossF1": return metricOf(r.result?.cross, "f1");
    case "inDomainAp": return metricOf(r.result?.inDomain, "ap");
    case "epochs": return r.epochs?.last ?? null;
    case "updated": {
      const t = r.files.newestMtime ? Date.parse(r.files.newestMtime) : NaN;
      return Number.isNaN(t) ? null : t;
    }
  }
}

/**
 * A sorted copy of `rows`. Missing values always come last, whichever the direction, and ties fall back to the
 * run name in ascending order so the table does not shuffle between refreshes.
 */
export function sortRuns(rows: readonly RunRow[], key: SortKey, dir: SortDir): RunRow[] {
  const sign = dir === "asc" ? 1 : -1;
  return [...rows].sort((a, b) => {
    const va = valueOf(a, key);
    const vb = valueOf(b, key);
    if (va === null && vb !== null) return 1;
    if (vb === null && va !== null) return -1;
    if (va !== null && vb !== null) {
      const c = typeof va === "number" && typeof vb === "number" ? va - vb : collator.compare(String(va), String(vb));
      if (c !== 0) return c * sign;
    }
    // Last resort, so the order is total even for names the collator treats as equal (case, leading zeros).
    return collator.compare(a.name, b.name) || (a.name < b.name ? -1 : a.name > b.name ? 1 : 0);
  });
}

export const SORT_LABEL: Record<SortKey, string> = {
  name: "Run",
  status: "Status",
  detector: "Held-out detector",
  crossAp: "Cross AP",
  crossF1: "Cross F1",
  inDomainAp: "In-domain AP",
  epochs: "Epochs",
  updated: "Updated",
};

/** The words read to a screen reader after a sort, so the direction is not only an arrow. */
export function describeSort(key: SortKey, dir: SortDir): string {
  const order = {
    status: dir === "asc" ? "needs attention first" : "complete first",
    time: dir === "desc" ? "newest first" : "oldest first",
    text: dir === "asc" ? "A to Z" : "Z to A",
    number: dir === "desc" ? "highest first" : "lowest first",
  }[KIND[key]];
  return `Sorted by ${SORT_LABEL[key]}, ${order}`;
}
