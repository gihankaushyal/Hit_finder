import { useState } from "react";
import { Bell } from "@phosphor-icons/react";

const BELL_ICON_SIZE = 15;
const LABEL = "Enable desktop notifications";

const canAsk = (): boolean => typeof Notification !== "undefined" && Notification.permission === "default";

/** Asks for desktop-notification permission on a click (browsers refuse an unprompted request). Hidden once decided. */
export function NotifyButton() {
  const [visible, setVisible] = useState(canAsk);
  if (!visible) return null;
  const ask = async () => {
    try {
      await Notification.requestPermission();
    } finally {
      setVisible(canAsk());
    }
  };
  return (
    <button type="button" className="btn btn--sm" aria-label={LABEL} title={LABEL} onClick={() => void ask()}>
      <Bell size={BELL_ICON_SIZE} aria-hidden="true" />
    </button>
  );
}
