import "./Badge.css";

/** Additions in green and deletions in red, as two pills. */
export function DiffChips({ additions, deletions }: { additions: number; deletions: number }) {
  return (
    <span className="diff-chips">
      <span className="diff-chip">+{additions}</span>
      <span className="diff-chip diff-chip--del">-{deletions}</span>
    </span>
  );
}
