import type { ReactNode } from "react";

/** `code` and **bold**. Anything else, including HTML, stays plain text: this returns React nodes only. */
const TOKEN = /(`[^`\n]+`|\*\*[^*\n]+\*\*)/g;

/** With a capture group, split puts the matched tokens at the odd indices and plain text at the even ones. */
export function renderInline(text: string): ReactNode[] {
  return text.split(TOKEN).flatMap((part, i): ReactNode[] => {
    if (part === "") return [];
    if (i % 2 === 0) return [part];
    if (part.startsWith("`")) return [<code key={i} className="mono">{part.slice(1, -1)}</code>];
    // Bold may hold a code span; the inner text cannot hold another bold, so this recursion ends.
    return [<strong key={i}>{renderInline(part.slice(2, -2))}</strong>];
  });
}
