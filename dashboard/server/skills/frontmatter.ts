export type FrontmatterValue = string | boolean;

const BLOCK_RE = /^---\r?\n([\s\S]*?)\r?\n---/;
const KEY_RE = /^([A-Za-z0-9_-]+):[ \t]*(.*)$/;
const BLOCK_SCALAR_RE = /^[>|][+-]?$/;
const TRAILING_COMMENT_RE = /\s+#.*$/;

function unquote(raw: string): string {
  if (raw.length >= 2 && raw.startsWith('"') && raw.endsWith('"')) return raw.slice(1, -1).replace(/\\(["\\])/g, "$1");
  if (raw.length >= 2 && raw.startsWith("'") && raw.endsWith("'")) return raw.slice(1, -1).replace(/''/g, "'");
  return raw;
}

/**
 * The top-level keys of a SKILL.md header, enough YAML for skill files: plain and quoted values, values continued
 * on indented lines, `>` and `|` block scalars, trailing comments and true/false. No nesting, lists or anchors.
 */
export function parseFrontmatter(text: string): Record<string, FrontmatterValue> {
  const block = BLOCK_RE.exec(text)?.[1];
  if (block === undefined) return {};
  const lines = block.split(/\r?\n/);
  const out: Record<string, FrontmatterValue> = {};
  for (let i = 0; i < lines.length; i++) {
    const m = KEY_RE.exec(lines[i]);
    if (!m) continue;
    const rest: string[] = [];
    while (i + 1 < lines.length && (/^[ \t]/.test(lines[i + 1]) || lines[i + 1] === "")) rest.push(lines[++i]);
    const more = rest.map((l) => l.trim());
    const head = m[2].trim();
    if (BLOCK_SCALAR_RE.test(head)) {
      out[m[1]] = more.join(head.startsWith(">") ? " " : "\n").replace(/\s+$/, "").replace(/^\s+/, "");
      continue;
    }
    const joined = [head, ...more].filter((l) => l !== "").join(" ");
    const quoted = joined.startsWith('"') || joined.startsWith("'");
    const value = quoted ? unquote(joined) : joined.replace(TRAILING_COMMENT_RE, "");
    out[m[1]] = !quoted && (value === "true" || value === "false") ? value === "true" : value;
  }
  return out;
}
