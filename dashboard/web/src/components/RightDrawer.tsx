import { useEffect, useRef } from "react";
import { X } from "@phosphor-icons/react";
import { Agents } from "../panels/Agents";
import { Approvals } from "../panels/Approvals";
import { ChiefChat } from "../panels/ChiefChat";
import "./RightDrawer.css";

const FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

/** Approvals, chief of staff and agents in a slide-over. It stays mounted while closed so drafts and logs survive. */
export function RightDrawer({ open, onClose }: { open: boolean; onClose: () => void }) {
  const closeButton = useRef<HTMLButtonElement>(null);
  const panel = useRef<HTMLElement>(null);
  useEffect(() => {
    if (!open) return;
    closeButton.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      // Escape in a native select dismisses its list, not the drawer.
      if (e.key === "Escape" && !e.defaultPrevented && !(e.target instanceof HTMLSelectElement)) onClose();
      if (e.key !== "Tab" || !panel.current) return;
      // The drawer is modal: Tab wraps inside it instead of reaching the page behind the scrim.
      const stops = Array.from(panel.current.querySelectorAll<HTMLElement>(FOCUSABLE)).filter((el) => {
        if (el.closest("[inert]")) return false;
        const style = getComputedStyle(el);
        return style.display !== "none" && style.visibility !== "hidden";
      });
      if (stops.length === 0) return;
      const first = stops[0];
      const last = stops[stops.length - 1];
      const here = document.activeElement;
      if (!panel.current.contains(here)) {
        e.preventDefault();
        first.focus();
      } else if (e.shiftKey && here === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && here === last) {
        e.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, onClose]);
  return (
    <>
      {/* Mouse-only backdrop: hidden from assistive tech so the modal boundary stays clean; the close button and Escape serve everyone else. */}
      <button type="button" className="scrim" data-open={open} aria-hidden="true" tabIndex={-1} onClick={onClose} />
      <aside ref={panel} id="agents-drawer" className="side-drawer" data-open={open} role="dialog" aria-modal="true" aria-label="Agents and chief of staff" inert={!open}>
        <header className="side-drawer__head">
          <h2>Agents and chief of staff</h2>
          <button ref={closeButton} type="button" className="btn btn--sm btn--icon" onClick={onClose} aria-label="Close agents panel">
            <X size={14} aria-hidden="true" />
          </button>
        </header>
        <div className="side-drawer__body">
          <Approvals />
          <ChiefChat />
          <Agents />
        </div>
      </aside>
    </>
  );
}
