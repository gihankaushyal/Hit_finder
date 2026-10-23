import { useState } from "react";
import { Moon, Sun } from "@phosphor-icons/react";
import { saveTheme, setThemeAttribute, storedTheme, systemTheme, type Theme } from "../lib/theme";

const ICON_SIZE = 16;

export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(() => storedTheme() ?? systemTheme());
  const toggle = () => {
    const next: Theme = theme === "light" ? "dark" : "light";
    setTheme(next);
    setThemeAttribute(next);
    saveTheme(next);
  };
  return (
    <button type="button" className="btn btn--sm btn--icon" aria-pressed={theme === "light"} onClick={toggle}>
      {theme === "light" ? <Sun size={ICON_SIZE} aria-hidden="true" /> : <Moon size={ICON_SIZE} aria-hidden="true" />}
      <span className="sr-only">Light theme</span>
    </button>
  );
}
