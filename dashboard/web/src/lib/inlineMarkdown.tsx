import type { ReactNode } from "react";

/** `code` and **bold**. Anything else, including HTML, stays plain text: this returns React nodes only. */
const TOKEN = /(`[^`\n]+`|\*\*[^*\n]+\*\*)/g;

export function renderInline(text: string): ReactNode[] {
  return text.split(TOKEN).flatMap((part, i): ReactNode[] => {
    if (part === "") return [];
    if (part.length > 2 && part.startsWith("`") && part.endsWith("`")) {
      return [<code key={i} className="mono">{part.slice(1, -1)}</code>];
    }
    if (part.length > 4 && part.startsWith("**") && part.endsWith("**")) {
      return [<strong key={i}>{part.slice(2, -2)}</strong>];
    }
    return [part];
  });
}
