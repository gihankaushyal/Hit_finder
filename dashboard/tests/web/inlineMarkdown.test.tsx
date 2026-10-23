import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import { renderInline } from "../../web/src/lib/inlineMarkdown";

const html = (text: string) => {
  const { container } = render(<p>{renderInline(text)}</p>);
  return container.firstElementChild as HTMLElement;
};

describe("renderInline", () => {
  it("turns backticked terms into code chips and double stars into bold", () => {
    const p = html("Writes `results.json` and keeps **best 187** epochs");
    expect(p.querySelector("code")?.textContent).toBe("results.json");
    expect(p.querySelector("strong")?.textContent).toBe("best 187");
    expect(p.textContent).toBe("Writes results.json and keeps best 187 epochs");
  });

  it("leaves plain text alone", () => {
    const p = html("nothing special here");
    expect(p.children).toHaveLength(0);
    expect(p.textContent).toBe("nothing special here");
  });

  it("never turns markup into elements, even inside code or bold", () => {
    const p = html('<img src=x onerror=alert(1)> `<script>x</script>` **<b>hi</b>**');
    expect(p.querySelector("img")).toBeNull();
    expect(p.querySelector("script")).toBeNull();
    expect(p.querySelector("b")).toBeNull();
    expect(p.querySelector("code")?.textContent).toBe("<script>x</script>");
    expect(p.querySelector("strong")?.textContent).toBe("<b>hi</b>");
  });

  it("keeps unmatched markers as literal text", () => {
    expect(html("a ` b and ** c").textContent).toBe("a ` b and ** c");
    expect(html("``").textContent).toBe("``");
  });

  it("handles several terms in a row", () => {
    const p = html("`a``b` **c****d**");
    expect(Array.from(p.querySelectorAll("code")).map((c) => c.textContent)).toEqual(["a", "b"]);
    expect(Array.from(p.querySelectorAll("strong")).map((c) => c.textContent)).toEqual(["c", "d"]);
  });

  it("renders a code span inside bold", () => {
    const p = html("**use `x` here**");
    expect(p.querySelector("strong code")?.textContent).toBe("x");
    expect(p.textContent).toBe("use x here");
  });

  it("does not turn plain text that merely starts and ends with backticks into code", () => {
    const p = html("`a\nb`");
    expect(p.querySelector("code")).toBeNull();
    expect(p.textContent).toBe("`a\nb`");
  });
});
