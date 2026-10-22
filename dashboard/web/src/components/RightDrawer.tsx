import { useEffect, useRef } from "react";
import { X } from "@phosphor-icons/react";
import { Agents } from "../panels/Agents";
import { Approvals } from "../panels/Approvals";
import { ChiefChat } from "../panels/ChiefChat";
import "./RightDrawer.css";

/** Approvals, chief of staff and agents in a slide-over. It stays mounted while closed so drafts and logs survive. */
export function RightDrawer({ open, onClose }: { open: boolean; onClose: () => void }) {
  const closeButton = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (!open) return;
    closeButton.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, onClose]);
  return (
    <>
      <button type="button" className="scrim" data-open={open} aria-label="Close agents panel" tabIndex={-1} onClick={onClose} />
      <aside id="agents-drawer" className="side-drawer" data-open={open} role="dialog" aria-label="Agents and chief of staff" inert={!open}>
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
