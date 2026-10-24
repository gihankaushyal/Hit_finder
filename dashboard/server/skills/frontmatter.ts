import { parse } from "yaml";

export type FrontmatterValue = string | boolean;

const BLOCK_RE = /^---\r?\n([\s\S]*?)\r?\n---/;

/**
 * The top-level text and true/false keys of a SKILL.md header (other values, nested keys and lists are skipped).
 * No header gives an empty object; a header that is not valid YAML throws.
 */
export function parseFrontmatter(text: string): Record<string, FrontmatterValue> {
  const block = BLOCK_RE.exec(text)?.[1];
  if (block === undefined) return {};
  const doc: unknown = parse(block);
  if (typeof doc !== "object" || doc === null || Array.isArray(doc)) return {};
  const out: Record<string, FrontmatterValue> = {};
  // A block scalar (`>` or `|`) ends with a newline in YAML; the callers want the text without it.
  for (const [key, value] of Object.entries(doc)) if (typeof value === "string" || typeof value === "boolean") out[key] = typeof value === "string" ? value.trim() : value;
  return out;
}
