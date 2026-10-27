import { useSyncExternalStore } from "react";
import { X } from "@phosphor-icons/react";
import { dismissToast, getToasts, subscribeToasts } from "../lib/attentionEffects";
import "./ToastHost.css";

const CLOSE_ICON_SIZE = 12;

/** Short-lived notices when an agent finishes or fails with the shelf closed. Not a live region: the nav bubble announces. */
export function ToastHost() {
  const toasts = useSyncExternalStore(subscribeToasts, getToasts, getToasts);
  if (toasts.length === 0) return null;
  return (
    <ul className="toasts">
      {toasts.map((t) => (
        <li key={t.id} className={`toast toast--${t.color}`}>
          <span>{t.text}</span>
          <button type="button" className="toast__close" aria-label={`Dismiss: ${t.text}`} onClick={() => dismissToast(t.id)}>
            <X size={CLOSE_ICON_SIZE} aria-hidden="true" />
          </button>
        </li>
      ))}
    </ul>
  );
}
