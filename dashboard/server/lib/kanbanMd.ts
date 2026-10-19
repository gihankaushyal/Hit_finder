export interface MdItem {
  kind: "item";
  checked: boolean;
  issue: number | null; // from "<!-- gh:#N -->" on the first line
  section: string; // text of the nearest preceding "## " heading, without "## "
  raw: string; // exact block text including its trailing newline
}
export interface MdText {
  kind: "text";
  raw: string;
}
export interface MdDoc {
  segments: (MdItem | MdText)[];
}

export const INBOX_HEADING = "Inbox (added via dashboard)";

const ITEM_START = /^- \[([ xX])\] /;
const MARKER = /<!-- gh:#(\d+) -->/;
const MARKER_WITH_SPACE = /[ \t]*<!-- gh:#\d+ -->/;
const BODY_DEDENT = 6;
const TITLE_CAP = 120;

// Splits into lines, each keeping its own terminator (\n or \r\n); the last
// line may have none. Concatenating the result reproduces the input exactly.
function splitLines(text: string): string[] {
  return text.match(/[^\n]*\n|[^\n]+$/g) ?? [];
}

const isBlank = (line: string): boolean => line.trim() === "";
const eolOf = (line: string): string => (line.endsWith("\r\n") ? "\r\n" : line.endsWith("\n") ? "\n" : "");
const stripEol = (line: string): string => line.replace(/\r?\n$/, "");

function headingOf(line: string): string | null {
  if (!line.startsWith("## ")) return null;
  return stripEol(line).slice(3);
}

function firstLineOf(raw: string): string {
  const nl = raw.indexOf("\n");
  return nl === -1 ? raw : raw.slice(0, nl + 1);
}

function docEol(doc: MdDoc): string {
  for (const s of doc.segments) if (s.raw.includes("\r\n")) return "\r\n";
  return "\n";
}

export function parseKanban(text: string): MdDoc {
  const lines = splitLines(text);
  const segments: (MdItem | MdText)[] = [];
  let section = "";
  let textBuf = "";
  const flushText = () => {
    if (textBuf) segments.push({ kind: "text", raw: textBuf });
    textBuf = "";
  };

  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    const m = ITEM_START.exec(line);
    if (!m) {
      const h = headingOf(line);
      if (h !== null) section = h;
      textBuf += line;
      i++;
      continue;
    }
    // Item: extend over blank and indented lines, then give trailing blanks back to text.
    let end = i + 1;
    while (end < lines.length && (isBlank(lines[end]) || /^[ \t]/.test(lines[end]))) end++;
    while (end > i + 1 && isBlank(lines[end - 1])) end--;
    flushText();
    const raw = lines.slice(i, end).join("");
    const marker = MARKER.exec(line);
    segments.push({
      kind: "item",
      checked: m[1] !== " ",
      issue: marker ? Number(marker[1]) : null,
      section,
      raw,
    });
    i = end;
  }
  flushText();
  return { segments };
}

export function serializeKanban(doc: MdDoc): string {
  return doc.segments.map((s) => s.raw).join("");
}

export function items(doc: MdDoc): MdItem[] {
  return doc.segments.filter((s): s is MdItem => s.kind === "item");
}

function firstLineText(item: MdItem): string {
  return stripEol(firstLineOf(item.raw)).replace(ITEM_START, "").replace(MARKER_WITH_SPACE, "");
}

export function itemTitle(item: MdItem): string {
  const first = firstLineText(item);
  let title: string;
  const bold = /^\*\*(.+?)\*\*/.exec(first);
  if (bold) {
    title = bold[1];
  } else {
    title = first.replace(/\*\*/g, "");
    const cuts = [title.indexOf(" — "), title.indexOf(" → ")].filter((n) => n >= 0);
    if (cuts.length) title = title.slice(0, Math.min(...cuts));
  }
  title = title.trim();
  return title.length > TITLE_CAP ? title.slice(0, TITLE_CAP - 3) + "..." : title;
}

export function itemBody(item: MdItem): string {
  const lines = item.raw.split(/\r?\n/);
  const first = firstLineText(item);
  const dedent = new RegExp(`^ {1,${BODY_DEDENT}}`);
  const rest = lines.slice(1).map((l) => l.replace(dedent, ""));
  return [first, ...rest].join("\n").trim();
}

export function setChecked(item: MdItem, checked: boolean): void {
  if (item.checked === checked) return; // keeps [X] as-is when nothing changes
  item.raw = item.raw.slice(0, 3) + (checked ? "x" : " ") + item.raw.slice(4);
  item.checked = checked;
}

export function setIssue(item: MdItem, n: number): void {
  const first = firstLineOf(item.raw);
  const tail = item.raw.slice(first.length);
  const marker = `<!-- gh:#${n} -->`;
  let newFirst: string;
  if (MARKER.test(first)) {
    newFirst = first.replace(MARKER, marker);
  } else {
    const eol = eolOf(first);
    newFirst = first.slice(0, first.length - eol.length) + " " + marker + eol;
  }
  item.raw = newFirst + tail;
  item.issue = n;
}

export function appendToInbox(doc: MdDoc, title: string, issue: number, checked: boolean): void {
  const eol = docEol(doc);
  const clean = title.replace(/\r?\n/g, " ");
  const newItem = (): MdItem => ({
    kind: "item",
    checked,
    issue,
    section: INBOX_HEADING,
    raw: `- [${checked ? "x" : " "}] ${clean} <!-- gh:#${issue} -->${eol}`,
  });
  const ensureNewline = (seg: MdItem | MdText) => {
    if (seg.raw && !seg.raw.endsWith("\n")) seg.raw += eol;
  };

  // Existing inbox with items: insert after the last one.
  const segs = doc.segments;
  let lastInboxItem = -1;
  segs.forEach((s, idx) => {
    if (s.kind === "item" && s.section === INBOX_HEADING) lastInboxItem = idx;
  });
  if (lastInboxItem >= 0) {
    ensureNewline(segs[lastInboxItem]);
    segs.splice(lastInboxItem + 1, 0, newItem());
    return;
  }

  // Existing inbox heading without items: insert just after the heading (and one blank line).
  for (let idx = 0; idx < segs.length; idx++) {
    const s = segs[idx];
    if (s.kind !== "text") continue;
    const lines = splitLines(s.raw);
    const h = lines.findIndex((l) => headingOf(l) === INBOX_HEADING);
    if (h < 0) continue;
    let cut = h + 1;
    if (cut < lines.length && isBlank(lines[cut]) && lines[cut].endsWith("\n")) cut++;
    const before = lines.slice(0, cut);
    const after = lines.slice(cut);
    if (before.length && !before[before.length - 1].endsWith("\n")) {
      before[before.length - 1] += eol;
    }
    const replacement: (MdItem | MdText)[] = [{ kind: "text", raw: before.join("") }, newItem()];
    if (after.length) replacement.push({ kind: "text", raw: after.join("") });
    segs.splice(idx, 1, ...replacement);
    return;
  }

  // No inbox section: create it at the end.
  if (segs.length) ensureNewline(segs[segs.length - 1]);
  segs.push({ kind: "text", raw: `${eol}## ${INBOX_HEADING}${eol}${eol}` });
  segs.push(newItem());
}
