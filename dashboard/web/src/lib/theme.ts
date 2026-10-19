export type Theme = "dark" | "light";
export const THEME_KEY = "dash-theme";

export function storedTheme(): Theme | null {
  try {
    const v = localStorage.getItem(THEME_KEY);
    return v === "dark" || v === "light" ? v : null;
  } catch {
    return null;
  }
}

export function saveTheme(t: Theme): void {
  try {
    localStorage.setItem(THEME_KEY, t);
  } catch {
    // storage blocked: the choice lasts for this page only
  }
}

export function systemTheme(): Theme {
  try {
    return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  } catch {
    return "dark";
  }
}

export const setThemeAttribute = (t: Theme): void => {
  document.documentElement.setAttribute("data-theme", t);
};

/** Runs before the first render so a saved choice does not flash the other theme. */
export function applyStoredTheme(): void {
  const t = storedTheme();
  if (t) setThemeAttribute(t);
}
