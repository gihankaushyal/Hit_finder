const MIN_S = 60;
const HOUR_S = 3600;
const DAY_S = 86400;

/** "3 min ago", "2 h ago", "5 d ago". An unparsable input gives an empty string. */
export function relTime(iso: string, now: Date = new Date()): string {
  const then = Date.parse(iso);
  if (Number.isNaN(then)) return "";
  const sec = Math.max(0, Math.floor((now.getTime() - then) / 1000));
  if (sec < MIN_S) return "just now";
  if (sec < HOUR_S) return `${Math.floor(sec / MIN_S)} min ago`;
  if (sec < DAY_S) return `${Math.floor(sec / HOUR_S)} h ago`;
  return `${Math.floor(sec / DAY_S)} d ago`;
}

/** "4m 12s", "12s", "1h 2m". */
export function duration(totalSec: number): string {
  const sec = Math.max(0, Math.round(totalSec));
  if (sec < MIN_S) return `${sec}s`;
  if (sec < HOUR_S) return `${Math.floor(sec / MIN_S)}m ${sec % MIN_S}s`;
  return `${Math.floor(sec / HOUR_S)}h ${Math.floor((sec % HOUR_S) / MIN_S)}m`;
}
