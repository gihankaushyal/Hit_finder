import { useEffect, useRef, useState } from "react";
import { Check, Copy } from "@phosphor-icons/react";
import "./CopyButton.css";

const COPIED_MS = 1600;
const ICON_SIZE = 13;

/** Copies text to the clipboard and says so. Quiet when the clipboard is blocked. */
export function CopyButton({ text, label }: { text: string; label: string }) {
  const [copied, setCopied] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout>>(undefined);
  useEffect(() => () => clearTimeout(timer.current), []);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      return;
    }
    setCopied(true);
    clearTimeout(timer.current);
    timer.current = setTimeout(() => setCopied(false), COPIED_MS);
  };
  return (
    <>
      <button type="button" className="copy-btn" aria-label={label} onClick={copy}>
        {copied ? <Check size={ICON_SIZE} aria-hidden="true" /> : <Copy size={ICON_SIZE} aria-hidden="true" />}
      </button>
      <span className="sr-only" role="status">{copied ? "Copied" : ""}</span>
    </>
  );
}
