import { describe, it, expect } from "vitest";
import fs from "node:fs";
import path from "node:path";
import * as K from "../server/lib/kanbanMd";

const sample = fs.readFileSync(path.join(__dirname, "fixtures/kanban-sample.md"), "utf8");
const realPath = path.join(__dirname, "../../phase-05-kanban.md");

/** Indices of lines that differ; asserts the line count is unchanged. */
function changedLines(a: string, b: string): number[] {
  const x = a.split("\n");
  const y = b.split("\n");
  expect(y.length).toBe(x.length);
  const out: number[] = [];
  x.forEach((l, i) => {
    if (l !== y[i]) out.push(i);
  });
  return out;
}

describe("kanbanMd", () => {
  it("round-trips byte for byte", () => {
    expect(K.serializeKanban(K.parseKanban(sample))).toBe(sample);
    const crlf = sample.replace(/\n/g, "\r\n");
    expect(K.serializeKanban(K.parseKanban(crlf))).toBe(crlf);
    expect(K.serializeKanban(K.parseKanban(""))).toBe("");
    expect(K.serializeKanban(K.parseKanban("- [ ] no trailing newline"))).toBe(
      "- [ ] no trailing newline",
    );
  });

  it("finds items with section, state and issue marker", () => {
    const its = K.items(K.parseKanban(sample));
    expect(its.map((i) => [i.section.split(" (")[0], i.checked, i.issue])).toEqual([
      ["Open decisions", true, null],
      ["Open decisions", false, null],
      ["In progress", false, null],
      ["In progress", true, 12],
      ["All tracked tasks", true, null],
      ["All tracked tasks", false, null],
    ]);
  });

  it("keeps the blank-line table inside its item", () => {
    const it2 = K.items(K.parseKanban(sample))[1];
    expect(it2.raw).toContain("| 1 | 0.9464 |");
    expect(it2.raw).toContain("Train loss drops monotonically.");
    expect(it2.raw.endsWith("monotonically.\n")).toBe(true);
  });

  it("derives titles", () => {
    const t = K.items(K.parseKanban(sample)).map(K.itemTitle);
    expect(t).toEqual([
      "Re-pretrain SSL folds 1–2?",
      "crops_per_frame>1 degrades val AP",
      "Risk B4",
      "Frame cache build-out",
      "Task 0: worktree setup",
      "Fold-2 A/B validation of frame cache",
    ]);
  });

  it("caps long titles at 120 characters", () => {
    const doc = K.parseKanban(`- [ ] ${"a".repeat(200)}\n`);
    const t = K.itemTitle(K.items(doc)[0]);
    expect(t.length).toBe(120);
    expect(t.endsWith("...")).toBe(true);
  });

  it("derives a dedented body without checkbox or marker", () => {
    const b = K.itemBody(K.items(K.parseKanban(sample))[2]);
    expect(b).toBe("Risk B4 — diagnose early stopping (lr 5e-5 peaks\nwithin ~2 epochs)");
    expect(K.itemBody(K.items(K.parseKanban(sample))[3])).toBe("Frame cache build-out");
  });

  it("setChecked changes exactly one character", () => {
    const doc = K.parseKanban(sample);
    K.setChecked(K.items(doc)[2], true);
    const out = K.serializeKanban(doc);
    expect(out.length).toBe(sample.length);
    expect(out).toContain("- [x] Risk B4");
    let diff = 0;
    for (let i = 0; i < out.length; i++) if (out[i] !== sample[i]) diff++;
    expect(diff).toBe(1);
  });

  it("setIssue appends a marker to the first line only", () => {
    const doc = K.parseKanban(sample);
    K.setIssue(K.items(doc)[2], 40);
    const out = K.serializeKanban(doc);
    expect(out).toContain("(lr 5e-5 peaks <!-- gh:#40 -->\n      within ~2 epochs)");
    expect(K.items(K.parseKanban(out))[2].issue).toBe(40);
  });

  it("appendToInbox creates the section once and appends items", () => {
    const doc = K.parseKanban(sample);
    K.appendToInbox(doc, "New task\nwith newline", 50, false);
    K.appendToInbox(doc, "Second", 51, true);
    const out = K.serializeKanban(doc);
    expect(out.startsWith(sample)).toBe(true);
    expect(out.match(/## Inbox \(added via dashboard\)/g)?.length).toBe(1);
    const its = K.items(K.parseKanban(out)).slice(-2);
    expect(its.map((i) => [i.issue, i.checked, K.itemTitle(i)])).toEqual([
      [50, false, "New task with newline"],
      [51, true, "Second"],
    ]);
    expect(its[0].section).toBe(K.INBOX_HEADING);
  });

  describe("what counts as an item", () => {
    it("only a column-0 '- [ ]' / '- [x]' / '- [X]' starts an item", () => {
      const doc = K.parseKanban(sample);
      expect(K.items(doc).length).toBe(6);
      const fold2 = K.items(doc)[5];
      // indented sub-items are continuation lines of their parent
      expect(fold2.raw).toContain("  - [ ] nested sub-bullet, not a task item\n");
      expect(fold2.raw).toContain("  - plain sub-bullet");
      // '* [ ]' at column 0 is plain text and ends the previous item
      const star = doc.segments.find((s) => s.kind === "text" && s.raw.includes("* [ ] star"));
      expect(star).toBeDefined();
    });

    it("treats text before the first heading as plain text with an empty section", () => {
      const doc = K.parseKanban("intro line\n- [ ] early item\n## H\n- [ ] later\n");
      const its = K.items(doc);
      expect(its.map((i) => i.section)).toEqual(["", "H"]);
      expect(doc.segments[0]).toEqual({ kind: "text", raw: "intro line\n" });
    });

    it("section tracks ## headings only, not ###", () => {
      const its = K.items(K.parseKanban(sample));
      expect(its[4].section).toBe("All tracked tasks (snapshot)");
    });

    it("does not treat '- [ ]' without a following space as an item", () => {
      expect(K.items(K.parseKanban("- [ ]\n- [x]no space\n")).length).toBe(0);
    });
  });

  describe("byte preservation edge cases", () => {
    it("reads capital X as checked and preserves its case unless flipped", () => {
      const text = "- [X] shouty item\n- [ ] other\n";
      const doc = K.parseKanban(text);
      const [a] = K.items(doc);
      expect(a.checked).toBe(true);
      K.setChecked(a, true); // no change requested
      expect(K.serializeKanban(doc)).toBe(text);
      K.setChecked(a, false);
      expect(K.serializeKanban(doc)).toBe("- [ ] shouty item\n- [ ] other\n");
      K.setChecked(a, true);
      expect(K.serializeKanban(doc)).toBe("- [x] shouty item\n- [ ] other\n");
    });

    it("round-trips CRLF and keeps CRLF on edits", () => {
      const crlf = sample.replace(/\n/g, "\r\n");
      const doc = K.parseKanban(crlf);
      expect(K.items(doc).length).toBe(6);
      expect(K.items(doc)[1].raw.endsWith("monotonically.\r\n")).toBe(true);
      expect(K.itemTitle(K.items(doc)[2])).toBe("Risk B4");
      expect(K.itemBody(K.items(doc)[2])).toBe(
        "Risk B4 — diagnose early stopping (lr 5e-5 peaks\nwithin ~2 epochs)",
      );
      K.setIssue(K.items(doc)[2], 7);
      K.appendToInbox(doc, "crlf task", 8, false);
      const out = K.serializeKanban(doc);
      expect(out).toContain("peaks <!-- gh:#7 -->\r\n      within");
      expect(out.replace(/\r\n/g, "")).not.toMatch(/[\r\n]/);
      expect(out.endsWith("- [ ] crlf task <!-- gh:#8 -->\r\n")).toBe(true);
    });

    it("no trailing newline: round-trips, and append is well formed", () => {
      const text = "## A\n\n- [ ] only item";
      const doc = K.parseKanban(text);
      expect(K.serializeKanban(doc)).toBe(text);
      K.setIssue(K.items(doc)[0], 3);
      expect(K.serializeKanban(doc)).toBe("## A\n\n- [ ] only item <!-- gh:#3 -->");
      K.appendToInbox(doc, "added", 4, false);
      const out = K.serializeKanban(doc);
      expect(out).toBe(
        "## A\n\n- [ ] only item <!-- gh:#3 -->\n\n## Inbox (added via dashboard)\n\n- [ ] added <!-- gh:#4 -->\n",
      );
      const its = K.items(K.parseKanban(out));
      expect(its.map((i) => i.issue)).toEqual([3, 4]);
    });

    it("appendToInbox on an empty document", () => {
      const doc = K.parseKanban("");
      K.appendToInbox(doc, "first", 1, true);
      const out = K.serializeKanban(doc);
      expect(out).toContain("## Inbox (added via dashboard)\n\n- [x] first <!-- gh:#1 -->\n");
      expect(K.items(K.parseKanban(out))[0].checked).toBe(true);
    });

    it("appendToInbox adds after existing inbox items, before later sections", () => {
      const text =
        "## Inbox (added via dashboard)\n\n- [ ] one <!-- gh:#1 -->\n\n## Later\n\n- [ ] x\n";
      const doc = K.parseKanban(text);
      K.appendToInbox(doc, "two", 2, false);
      expect(K.serializeKanban(doc)).toBe(
        "## Inbox (added via dashboard)\n\n- [ ] one <!-- gh:#1 -->\n- [ ] two <!-- gh:#2 -->\n\n## Later\n\n- [ ] x\n",
      );
    });

    it("appendToInbox into an existing empty inbox section", () => {
      const text = "## Inbox (added via dashboard)\n\n## Later\n- [ ] x\n";
      const doc = K.parseKanban(text);
      K.appendToInbox(doc, "one", 1, false);
      expect(K.serializeKanban(doc)).toBe(
        "## Inbox (added via dashboard)\n\n- [ ] one <!-- gh:#1 -->\n## Later\n- [ ] x\n",
      );
      expect(K.items(K.parseKanban(K.serializeKanban(doc))).map((i) => i.section)).toEqual([
        K.INBOX_HEADING,
        "Later",
      ]);
    });
  });

  describe("single-line mutations", () => {
    it("setIssue on an item with a marker replaces it", () => {
      const doc = K.parseKanban(sample);
      const item = K.items(doc)[3];
      expect(item.issue).toBe(12);
      K.setIssue(item, 77);
      const out = K.serializeKanban(doc);
      expect(out).toContain("- [x] Frame cache build-out <!-- gh:#77 -->\n");
      expect(out).not.toContain("gh:#12");
      expect(out.match(/gh:#/g)?.length).toBe(2); // #77 and the sub-bullet #99
      expect(item.issue).toBe(77);
      expect(K.items(K.parseKanban(out))[3].issue).toBe(77);
    });

    it("setChecked and setIssue change exactly one line", () => {
      for (let idx = 0; idx < 6; idx++) {
        const a = K.parseKanban(sample);
        K.setChecked(K.items(a)[idx], !K.items(a)[idx].checked);
        expect(changedLines(sample, K.serializeKanban(a)).length).toBe(1);
        const b = K.parseKanban(sample);
        K.setIssue(K.items(b)[idx], 100 + idx);
        expect(changedLines(sample, K.serializeKanban(b)).length).toBe(1);
      }
    });

    it("setChecked to the current state is a no-op", () => {
      const doc = K.parseKanban(sample);
      K.setChecked(K.items(doc)[0], true);
      K.setChecked(K.items(doc)[1], false);
      expect(K.serializeKanban(doc)).toBe(sample);
    });

    it("marker-like text in body lines is not the marker", () => {
      const doc = K.parseKanban(sample);
      const fold2 = K.items(doc)[5];
      expect(fold2.raw).toContain("<!-- gh:#99 -->");
      expect(fold2.issue).toBeNull();
      K.setIssue(fold2, 5);
      const out = K.serializeKanban(doc);
      expect(out).toContain("- [ ] Fold-2 A/B validation of frame cache <!-- gh:#5 -->\n");
      expect(out).toContain("  - plain sub-bullet <!-- gh:#99 -->\n");
      expect(K.items(K.parseKanban(out))[5].issue).toBe(5);
      // a marker only on a continuation line must not be stripped from the body
      expect(K.itemBody(K.items(K.parseKanban(sample))[5])).toContain("<!-- gh:#99 -->");
    });
  });

  it("random mutation sequences leave every untouched item and text segment byte-identical", () => {
    let seed = 12345;
    const rnd = () => {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      return seed / 2 ** 32;
    };
    const orig = K.parseKanban(sample);
    const origItems = K.items(orig).map((i) => i.raw);
    const origTexts = orig.segments.filter((s) => s.kind === "text").map((s) => s.raw);

    for (let round = 0; round < 200; round++) {
      const doc = K.parseKanban(sample);
      const its = K.items(doc);
      const touched = new Set<number>();
      const steps = 1 + Math.floor(rnd() * 8);
      let appended = 0;
      for (let s = 0; s < steps; s++) {
        const op = Math.floor(rnd() * 3);
        const idx = Math.floor(rnd() * its.length);
        if (op === 0) {
          K.setChecked(its[idx], rnd() < 0.5);
          touched.add(idx);
        } else if (op === 1) {
          K.setIssue(its[idx], Math.floor(rnd() * 1000));
          touched.add(idx);
        } else {
          K.appendToInbox(doc, `task ${s}`, 2000 + s, rnd() < 0.5);
          appended++;
        }
      }
      const out = K.serializeKanban(doc);
      const reparsed = K.parseKanban(out);
      const reItems = K.items(reparsed);
      expect(reItems.length).toBe(origItems.length + appended);
      origItems.forEach((raw, i) => {
        if (!touched.has(i)) expect(reItems[i].raw).toBe(raw);
        else expect(changedLines(raw, reItems[i].raw).length).toBeLessThanOrEqual(1);
      });
      const reTexts = reparsed.segments.filter((s) => s.kind === "text").map((s) => s.raw);
      // the appended Inbox heading merges into the final original text segment on re-parse
      const last = origTexts.length - 1;
      expect(reTexts.slice(0, last)).toEqual(origTexts.slice(0, last));
      expect(reTexts[last].startsWith(origTexts[last])).toBe(true);
    }
  });

  it.skipIf(!fs.existsSync(realPath))(
    "round-trips the real kanban file byte for byte (read only)",
    () => {
      const text = fs.readFileSync(realPath, "utf8");
      const doc = K.parseKanban(text);
      expect(K.serializeKanban(doc)).toBe(text);
      expect(K.items(doc).length).toBeGreaterThan(50);
    },
  );
});
