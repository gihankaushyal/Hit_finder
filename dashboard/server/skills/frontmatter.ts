import { parse } from "yaml";

const FENCE = "---";
const BOM = "\uFEFF";

/**
 * The top-level text keys of a SKILL.md header (numbers, booleans, nested keys and lists are skipped).
 * No header (or an unclosed one) gives an empty object; a header that is not valid YAML throws.
 */
export function parseFrontmatter(text: string): Record<string, string> {
  const lines = text.replace(BOM, "").split(/\r?\n/);
  if (lines[0]?.trimEnd() !== FENCE) return {};
  const end = lines.findIndex((l, i) => i > 0 && l.trimEnd() === FENCE); // a line that only starts with --- does not close the header
  if (end < 0) return {};
  const doc: unknown = parse(lines.slice(1, end).join("\n"));
  if (typeof doc !== "object" || doc === null || Array.isArray(doc)) return {};
  const out: Record<string, string> = {};
  // A block scalar (`>` or `|`) ends with a newline in YAML; the callers want the text without it.
  for (const [key, value] of Object.entries(doc)) if (typeof value === "string") out[key] = value.trim();
  return out;
}
