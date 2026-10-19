import { describe, it, expect } from "vitest";
import fs from "node:fs";
import path from "node:path";

const CSS = fs.readFileSync(path.resolve(import.meta.dirname, "../../web/src/tokens.css"), "utf8");
const MIN_RATIO = 4.5; // WCAG AA, normal text

/** Variables declared inside the first block whose selector contains `selector`. */
function tokens(selector: string): Record<string, string> {
  const start = CSS.indexOf(selector);
  if (start < 0) throw new Error(`no ${selector} block in tokens.css`);
  const open = CSS.indexOf("{", start);
  const close = CSS.indexOf("}", open);
  const out: Record<string, string> = {};
  for (const m of CSS.slice(open, close).matchAll(/--([a-z0-9-]+)\s*:\s*(#[0-9a-fA-F]{6})\s*;/g)) out[m[1]] = m[2];
  return out;
}
function luminance(hex: string): number {
  const ch = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
  const [r, g, b] = ch.map((c) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}
export function ratio(a: string, b: string): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}

const THEMES: [string, Record<string, string>][] = [
  ["dark", tokens(":root {")],
  ["light", tokens(':root[data-theme="light"]')],
];
const FOREGROUNDS = ["text", "text-dim", "accent", "fail", "ok"];
const BACKGROUNDS = ["bg", "surface"];

describe("colour contrast (WCAG AA, at least 4.5:1)", () => {
  it("the media-query light block repeats the same values as the explicit light theme", () => {
    const media = tokens(":root:not([data-theme])");
    expect(media).toEqual(THEMES[1][1]);
  });

  for (const [name, t] of THEMES) {
    describe(name, () => {
      for (const fg of FOREGROUNDS) {
        for (const bg of BACKGROUNDS) {
          it(`${fg} on ${bg}`, () => {
            expect(ratio(t[fg], t[bg]), `${fg} ${t[fg]} on ${bg} ${t[bg]}`).toBeGreaterThanOrEqual(MIN_RATIO);
          });
        }
      }
      it("text and text-dim on surface-2", () => {
        expect(ratio(t["text"], t["surface-2"])).toBeGreaterThanOrEqual(MIN_RATIO);
        expect(ratio(t["text-dim"], t["surface-2"])).toBeGreaterThanOrEqual(MIN_RATIO);
      });
      it("the primary button text on the accent", () => {
        expect(ratio(t["accent-text"], t["accent"])).toBeGreaterThanOrEqual(MIN_RATIO);
      });
    });
  }

  it("prints the computed ratios", () => {
    const rows: string[] = [];
    for (const [name, t] of THEMES) {
      for (const fg of FOREGROUNDS) for (const bg of BACKGROUNDS) rows.push(`${name} ${fg} on ${bg}: ${ratio(t[fg], t[bg]).toFixed(2)}`);
      rows.push(`${name} accent-text on accent: ${ratio(t["accent-text"], t["accent"]).toFixed(2)}`);
    }
    console.log(rows.join("\n"));
    expect(rows.length).toBeGreaterThan(0);
  });
});
