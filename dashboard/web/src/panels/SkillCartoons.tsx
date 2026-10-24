import type { ReactElement } from "react";

/** Small flat cartoons, one per project skill. Colours come from the theme tokens so both themes read well. */
const STROKE = 2.5;

function Frame({ children }: { children: ReactElement | ReactElement[] }) {
  return (
    <svg className="skill-art" viewBox="0 0 64 64" fill="none" strokeWidth={STROKE} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" focusable="false">
      {children}
    </svg>
  );
}

/** docs-refresh: a stack of pages with a magnifier over it. */
function DocsArt() {
  return (
    <Frame>
      <rect x="10" y="8" width="30" height="40" rx="3" fill="var(--surface-2)" stroke="var(--text-faint)" />
      <rect x="15" y="13" width="30" height="40" rx="3" fill="var(--surface)" stroke="var(--text-dim)" />
      <path d="M21 23h18M21 30h18M21 37h10" stroke="var(--accent)" />
      <circle cx="44" cy="42" r="8" fill="var(--surface)" stroke="var(--ok)" />
      <path d="M50 48l7 7" stroke="var(--ok)" strokeWidth={STROKE + 1} />
      <path d="M41 42l2 2 4-4" stroke="var(--ok)" />
    </Frame>
  );
}

/** pipeline-check: a detector grid with a spark where a Bragg peak lands. */
function PipelineArt() {
  return (
    <Frame>
      <rect x="8" y="10" width="48" height="44" rx="4" fill="var(--surface)" stroke="var(--text-dim)" />
      <path d="M24 10v44M40 10v44M8 25h48M8 40h48" stroke="var(--rule-strong)" />
      <circle cx="16" cy="17" r="2" fill="var(--text-faint)" stroke="none" />
      <circle cx="48" cy="46" r="2" fill="var(--text-faint)" stroke="none" />
      <circle cx="32" cy="32" r="5" fill="var(--warn)" stroke="none" />
      <path d="M32 21v5M32 38v5M21 32h5M38 32h5M24 24l3 3M37 37l3 3M40 24l-3 3M27 37l-3 3" stroke="var(--warn)" />
    </Frame>
  );
}

/** slurm-job: a server rack launching a rocket. */
function SlurmArt() {
  return (
    <Frame>
      <rect x="8" y="30" width="30" height="10" rx="2" fill="var(--surface)" stroke="var(--text-dim)" />
      <rect x="8" y="43" width="30" height="10" rx="2" fill="var(--surface)" stroke="var(--text-dim)" />
      <circle cx="14" cy="35" r="1.6" fill="var(--ok)" stroke="none" />
      <circle cx="14" cy="48" r="1.6" fill="var(--ok)" stroke="none" />
      <path d="M22 35h10M22 48h10" stroke="var(--text-faint)" />
      <path d="M46 8c6 5 8 13 6 22l-6 4-6-4c-2-9 0-17 6-22z" fill="var(--surface-2)" stroke="var(--accent)" />
      <circle cx="46" cy="21" r="3" fill="var(--info)" stroke="none" />
      <path d="M40 30l-5 6M52 30l5 6" stroke="var(--accent)" />
      <path d="M43 38c0 5 1 8 3 11 2-3 3-6 3-11" fill="var(--warn)" stroke="none" />
    </Frame>
  );
}

/** A generic flask for any other skill folder that appears later. */
function GenericArt() {
  return (
    <Frame>
      <path d="M26 10h12M29 10v16L16 50a4 4 0 003.5 6h25a4 4 0 003.5-6L35 26V10" fill="var(--surface)" stroke="var(--text-dim)" />
      <path d="M21 42h22" stroke="var(--accent)" />
      <circle cx="30" cy="48" r="1.8" fill="var(--ok)" stroke="none" />
      <circle cx="37" cy="46" r="1.4" fill="var(--info)" stroke="none" />
    </Frame>
  );
}

const ART: Record<string, () => ReactElement> = {
  "docs-refresh": DocsArt,
  "pipeline-check": PipelineArt,
  "slurm-job": SlurmArt,
};

export function SkillArt({ name }: { name: string }) {
  const Art = ART[name] ?? GenericArt;
  return <Art />;
}
