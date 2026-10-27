import type { AgentState } from "../../../shared/types";
import type { BubbleColor } from "./attention";

export const TOAST_MS = 6000;
const FAVICON_SIZE = 32;
const FAVICON_DOT_RADIUS = 9;
const FAVICON_DOT_INSET = 2;
const TOKEN: Record<BubbleColor, string> = { red: "--fail", yellow: "--warn", green: "--ok" };

export function outcomeText(name: string, state: AgentState): string {
  if (state === "failed") return `${name} failed`;
  if (state === "idle") return `${name} finished a turn`;
  return `${name} finished`;
}

// ---- toasts: a tiny external store read by <ToastHost/> ----
export interface Toast {
  id: number;
  text: string;
  color: BubbleColor;
}
let toasts: Toast[] = [];
let nextId = 1;
const listeners = new Set<() => void>();
const emit = () => listeners.forEach((l) => l());

export const getToasts = (): Toast[] => toasts;
export function subscribeToasts(l: () => void): () => void {
  listeners.add(l);
  return () => listeners.delete(l);
}
export function dismissToast(id: number): void {
  toasts = toasts.filter((t) => t.id !== id);
  emit();
}
export function clearToasts(): void {
  toasts = [];
  emit();
}
export function pushToast(text: string, color: BubbleColor): void {
  const id = nextId++;
  toasts = [...toasts, { id, text, color }];
  emit();
  setTimeout(() => dismissToast(id), TOAST_MS);
}

// ---- browser Notification: only for a hidden tab, only after the user granted permission ----
export function notifyFinished(name: string, state: AgentState): void {
  try {
    if (typeof Notification === "undefined" || Notification.permission !== "granted" || !document.hidden) return;
    new Notification(outcomeText(name, state), { tag: `agent-${name}` });
  } catch {
    /* a notification is a courtesy; never break the page for it */
  }
}

// ---- tab title and favicon ----
let baseTitle: string | null = null;
let baseIcon: string | null = null;

function iconLink(): HTMLLinkElement {
  let link = document.querySelector<HTMLLinkElement>("link[rel~='icon']");
  if (!link) {
    link = document.createElement("link");
    link.rel = "icon";
    document.head.appendChild(link);
  }
  return link;
}

function drawDot(color: BubbleColor): string | null {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = FAVICON_SIZE;
  const ctx = canvas.getContext?.("2d");
  if (!ctx) return null;
  ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue(TOKEN[color]).trim() || color;
  ctx.beginPath();
  ctx.arc(FAVICON_SIZE - FAVICON_DOT_RADIUS - FAVICON_DOT_INSET, FAVICON_DOT_RADIUS + FAVICON_DOT_INSET, FAVICON_DOT_RADIUS, 0, Math.PI * 2);
  ctx.fill();
  return canvas.toDataURL("image/png");
}

/** `count` unseen results in the title; a coloured dot on the favicon. 0 / null restores both. */
export function setTabAttention(count: number, color: BubbleColor | null): void {
  try {
    if (baseTitle === null) baseTitle = document.title;
    document.title = count > 0 ? `(${count}) ${baseTitle}` : baseTitle;
    if (count === 0) baseTitle = null;
    const link = iconLink();
    if (color && count > 0) {
      if (baseIcon === null) baseIcon = link.getAttribute("href") ?? "";
      const dot = drawDot(color);
      if (dot) link.href = dot;
    } else if (baseIcon !== null) {
      if (baseIcon) link.href = baseIcon;
      else link.remove();
      baseIcon = null;
    }
  } catch {
    /* best effort */
  }
}
