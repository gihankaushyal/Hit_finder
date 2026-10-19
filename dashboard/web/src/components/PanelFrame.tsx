import { useId, type ReactNode } from "react";

export interface PanelFrameProps {
  title: string;
  count?: number;
  actions?: ReactNode;
  state: "loading" | "error" | "ready";
  error?: string;
  onRetry?: () => void;
  children?: ReactNode;
}

const SKELETON_ROWS = 4;

export function PanelFrame({ title, count, actions, state, error, onRetry, children }: PanelFrameProps) {
  const headingId = useId();
  return (
    <section className="panel" aria-labelledby={headingId}>
      <header className="panel__head">
        <h2 id={headingId} className="panel__title">{title}</h2>
        {count !== undefined && <span className="mono panel__count">{count}</span>}
        {actions && <div className="panel__actions">{actions}</div>}
      </header>
      <div className="panel__body">
        {state === "loading" && (
          <>
            <span className="sr-only">Loading {title}</span>
            <div className="skeleton" aria-hidden="true">
              {Array.from({ length: SKELETON_ROWS }, (_, i) => (
                <div key={i} className="skeleton__row" />
              ))}
            </div>
          </>
        )}
        {state === "error" && (
          <div className="panel__error" role="alert">
            <p>{error ?? "Something went wrong."}</p>
            {onRetry && (
              <button type="button" className="btn" onClick={onRetry}>
                Retry
              </button>
            )}
          </div>
        )}
        {state === "ready" && children}
      </div>
    </section>
  );
}
