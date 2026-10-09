import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ThemeToggle } from "../../web/src/components/ThemeToggle";
import { THEME_KEY, applyStoredTheme } from "../../web/src/lib/theme";

function mockPrefersLight(light: boolean) {
  vi.stubGlobal("matchMedia", (q: string) => ({ matches: q.includes("light") ? light : false, media: q, addEventListener() {}, removeEventListener() {} }));
}
beforeEach(() => {
  localStorage.clear();
  document.documentElement.removeAttribute("data-theme");
});
afterEach(() => vi.unstubAllGlobals());

describe("ThemeToggle", () => {
  it("follows prefers-color-scheme until the user chooses, and the toggle persists the choice", async () => {
    mockPrefersLight(true);
    render(<ThemeToggle />);
    const btn = screen.getByRole("button", { name: /light theme/i });
    expect(btn).toHaveAttribute("aria-pressed", "true");
    expect(document.documentElement).not.toHaveAttribute("data-theme");
    await userEvent.click(btn);
    expect(document.documentElement).toHaveAttribute("data-theme", "dark");
    expect(localStorage.getItem(THEME_KEY)).toBe("dark");
    expect(btn).toHaveAttribute("aria-pressed", "false");
    await userEvent.click(btn);
    expect(document.documentElement).toHaveAttribute("data-theme", "light");
    expect(localStorage.getItem(THEME_KEY)).toBe("light");
  });

  it("dark by default when the system has no preference", () => {
    mockPrefersLight(false);
    render(<ThemeToggle />);
    expect(screen.getByRole("button", { name: /light theme/i })).toHaveAttribute("aria-pressed", "false");
  });

  it("applyStoredTheme restores the saved choice", () => {
    localStorage.setItem(THEME_KEY, "light");
    applyStoredTheme();
    expect(document.documentElement).toHaveAttribute("data-theme", "light");
  });

  it("still works when localStorage throws", async () => {
    mockPrefersLight(false);
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("denied");
    });
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("denied");
    });
    render(<ThemeToggle />);
    await userEvent.click(screen.getByRole("button", { name: /light theme/i }));
    expect(document.documentElement).toHaveAttribute("data-theme", "light");
    vi.restoreAllMocks();
  });
});
