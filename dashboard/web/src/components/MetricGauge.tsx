import "./MetricGauge.css";

/** At or above this a metric reads as healthy (green); below it the bar turns amber. */
export const METRIC_GOOD_MIN = 0.8;
const NA = "n/a";
const DECIMALS = 3;

/** A number with a small bar beside it. The bar only repeats the number, so it is hidden from assistive tech. */
export function MetricGauge({ value }: { value: number | null | undefined }) {
  if (typeof value !== "number" || Number.isNaN(value)) return <span className="gauge gauge--na mono">{NA}</span>;
  const fill = Math.min(1, Math.max(0, value));
  return (
    <span className="gauge">
      <span className="gauge__value mono">{value.toFixed(DECIMALS)}</span>
      <span className="gauge__bar" aria-hidden="true" data-level={value >= METRIC_GOOD_MIN ? "good" : "low"}>
        <span className="gauge__fill" style={{ transform: `scaleX(${fill})` }} />
      </span>
    </span>
  );
}
