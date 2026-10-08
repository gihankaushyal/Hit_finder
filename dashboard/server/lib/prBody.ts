export interface ParsedPrBody {
  summary: string[];
  testPlan: { text: string; checked: boolean }[];
}

const HEADING = /^#{1,6}\s+(.*?)\s*#*\s*$/;
const BULLET = /^\s{0,3}[-*+]\s+(.*)$/;
const CHECKBOX = /^\[([ xX])\]\s+(.*)$/;

/** Split a markdown body into lowercase-heading -> lines. Text before any heading is under "". */
function sections(body: string): Map<string, string[]> {
  const out = new Map<string, string[]>([["", []]]);
  let current = "";
  for (const line of body.split(/\r?\n/)) {
    const h = HEADING.exec(line);
    if (h) {
      current = h[1].trim().toLowerCase();
      if (!out.has(current)) out.set(current, []);
      continue;
    }
    out.get(current)!.push(line);
  }
  return out;
}

/** Bullets with indented continuation lines folded into the preceding bullet. */
function bullets(lines: string[]): string[] {
  const items: string[] = [];
  for (const line of lines) {
    const b = BULLET.exec(line);
    if (b) {
      items.push(b[1].trim());
    } else if (line.trim() && /^\s+/.test(line) && items.length > 0) {
      items[items.length - 1] += " " + line.trim();
    }
  }
  return items;
}

function firstParagraph(lines: string[]): string | null {
  const para: string[] = [];
  for (const line of lines) {
    if (!line.trim()) {
      if (para.length) break;
      continue;
    }
    para.push(line.trim());
  }
  return para.length ? para.join(" ") : null;
}

export function parsePrBody(body: string): ParsedPrBody {
  const secs = sections(body ?? "");
  const summaryLines = secs.get("summary");
  let summary: string[];
  if (summaryLines) {
    summary = bullets(summaryLines);
    if (summary.length === 0) {
      const p = firstParagraph(summaryLines);
      summary = p ? [p] : [];
    }
  } else {
    const p = firstParagraph(secs.get("") ?? []);
    summary = p ? [p] : [];
  }
  const testPlan = bullets(secs.get("test plan") ?? []).map((text) => {
    const c = CHECKBOX.exec(text);
    return c ? { text: c[2].trim(), checked: c[1].toLowerCase() === "x" } : { text, checked: false };
  });
  return { summary, testPlan };
}
