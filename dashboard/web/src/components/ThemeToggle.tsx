import { useState } from "react";
import { Moon, Sun } from "@phosphor-icons/react";
import { saveTheme, setThemeAttribute, storedTheme, systemTheme, type Theme } from "../lib/theme";

export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(() => storedTheme() ?? systemTheme());
  const toggle = () => {
    const next: Theme = theme === "light" ? "dark" : "light";
    setTheme(next);
    setThemeAttribute(next);
    saveTheme(next);
  };
  return (
    <button type="button" className="btn btn--icon" aria-pressed={theme === "light"} onClick={toggle}>
      {theme === "light" ? <Sun size={16} aria-hidden="true" /> : <Moon size={16} aria-hidden="true" />}
      <span>Light theme</span>
    </button>
  );
}
