import type { ReactNode } from "react";
import "./Badge.css";

export type BadgeTone = "neutral" | "ok" | "fail" | "warn" | "info" | "merged" | "open";

/** A small pill for a state. Pair the colour with an icon or a word; colour alone is not enough. */
export function Badge({ tone = "neutral", icon, children }: { tone?: BadgeTone; icon?: ReactNode; children: ReactNode }) {
  return (
    <span className="badge" data-tone={tone}>
      {icon}
      <span>{children}</span>
    </span>
  );
}
