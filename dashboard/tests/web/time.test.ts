import { describe, it, expect } from "vitest";
import { relTime, duration } from "../../web/src/lib/time";

const NOW = new Date("2026-10-08T12:00:00Z");
const ago = (sec: number): string => new Date(NOW.getTime() - sec * 1000).toISOString();

describe("relTime", () => {
  it.each([
    [10, "just now"],
    [3 * 60, "3 min ago"],
    [59 * 60, "59 min ago"],
    [2 * 3600, "2 h ago"],
    [23 * 3600, "23 h ago"],
    [5 * 86400, "5 d ago"],
  ])("%i seconds ago is %s", (sec, text) => {
    expect(relTime(ago(sec), NOW)).toBe(text);
  });
  it("treats a time in the future as just now and bad input as an empty string", () => {
    expect(relTime(new Date(NOW.getTime() + 5000).toISOString(), NOW)).toBe("just now");
    expect(relTime("not a date", NOW)).toBe("");
  });
});

describe("duration", () => {
  it.each([
    [0, "0s"],
    [12, "12s"],
    [252, "4m 12s"],
    [3725, "1h 2m"],
  ])("%i seconds is %s", (sec, text) => {
    expect(duration(sec)).toBe(text);
  });
});
