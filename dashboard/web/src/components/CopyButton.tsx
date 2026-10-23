import { useEffect, useRef, useState } from "react";
import { Check, Copy, X } from "@phosphor-icons/react";
import "./CopyButton.css";

const COPIED_MS = 1600;
const ICON_SIZE = 13;

/** Copies with the async clipboard, falling back to a hidden textarea where it is unavailable (plain http). */
async function writeClipboard(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    const before = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const field = document.createElement("textarea");
    field.value = text;
    field.setAttribute("readonly", "");
    field.style.cssText = "position:fixed;top:0;left:0;opacity:0";
    document.body.appendChild(field);
    field.select();
    try {
      return document.execCommand("copy");
    } catch {
      return false;
    } finally {
      field.remove();
      before?.focus();
    }
  }
}

/** Copies text to the clipboard and says so, including when it could not. */
export function CopyButton({ text, label }: { text: string; label: string }) {
  const [result, setResult] = useState<"copied" | "failed" | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout>>(undefined);
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      clearTimeout(timer.current);
    };
  }, []);
  const copy = async () => {
    const ok = await writeClipboard(text);
    // The write is async; the button may be gone by now.
    if (!mounted.current) return;
    setResult(ok ? "copied" : "failed");
    clearTimeout(timer.current);
    timer.current = setTimeout(() => setResult(null), COPIED_MS);
  };
  return (
    <>
      <button type="button" className="copy-btn" aria-label={label} onClick={copy}>
        {result === "copied" ? <Check size={ICON_SIZE} aria-hidden="true" /> : result === "failed" ? <X size={ICON_SIZE} aria-hidden="true" /> : <Copy size={ICON_SIZE} aria-hidden="true" />}
      </button>
      <span className="sr-only" role="status">{result === "copied" ? "Copied" : result === "failed" ? "Copy failed" : ""}</span>
    </>
  );
}
