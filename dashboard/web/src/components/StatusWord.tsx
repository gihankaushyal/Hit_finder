import type { ReactNode } from "react";
import { CheckCircle, CircleNotch, MinusCircle, WarningCircle, XCircle } from "@phosphor-icons/react";

export type StatusKind = "ok" | "fail" | "running" | "idle" | "attention";

const ICONS = { ok: CheckCircle, fail: XCircle, running: CircleNotch, idle: MinusCircle, attention: WarningCircle };
const ICON_SIZE = 14;

/** An icon plus a word, so state is never carried by colour alone. */
export function StatusWord({ kind, children }: { kind: StatusKind; children: ReactNode }) {
  const Icon = ICONS[kind];
  return (
    <span className="status-word" data-kind={kind}>
      <Icon size={ICON_SIZE} weight="bold" aria-hidden="true" className={kind === "running" ? "spin" : undefined} />
      <span>{children}</span>
    </span>
  );
}
