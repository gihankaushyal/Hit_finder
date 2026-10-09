import type { ReactNode } from "react";
import type { ResourceState } from "../lib/useResource";

/** GitHub text is untrusted: only plain web links become anchors. */
export function isWebUrl(url: string): boolean {
  return /^https?:\/\//i.test(url);
}

export function ExternalLink({ href, children, className }: { href: string; children: ReactNode; className?: string }) {
  if (!isWebUrl(href)) return <span className={className}>{children}</span>;
  return (
    <a href={href} target="_blank" rel="noopener noreferrer" className={className}>
      {children}
    </a>
  );
}

/** A panel shows its error state only when there is nothing to show; a failed refresh keeps the old data. */
export function panelState(r: ResourceState<unknown>): "loading" | "error" | "ready" {
  if (r.data === null) return r.error ? "error" : "loading";
  return "ready";
}

export function StaleNotice({ res, label }: { res: ResourceState<unknown>; label: string }) {
  if (!res.error || res.data === null) return null;
  return (
    <div className="notice notice--fail" role="alert">
      <span>Could not refresh {label}: {res.error}</span>
      <button type="button" className="btn" onClick={res.reload}>Retry</button>
    </div>
  );
}

/** Loading or error state for one part of a panel. */
export function SectionState({ res, label }: { res: ResourceState<unknown>; label: string }) {
  if (res.data === null && res.error) {
    return (
      <div className="panel__error" role="alert">
        <p>{res.error}</p>
        <button type="button" className="btn" onClick={res.reload} aria-label={`Retry ${label}`}>Retry</button>
      </div>
    );
  }
  return (
    <>
      <span className="sr-only">Loading {label}</span>
      <div className="skeleton" aria-hidden="true">
        <div className="skeleton__row" />
        <div className="skeleton__row" />
        <div className="skeleton__row" />
      </div>
    </>
  );
}

export function absoluteTime(iso: string): string {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString("en-GB", { dateStyle: "medium", timeStyle: "short" });
}
