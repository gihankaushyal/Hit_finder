import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { MetricGauge, METRIC_GOOD_MIN } from "../../web/src/components/MetricGauge";

const bar = (c: HTMLElement) => c.querySelector(".gauge__bar") as HTMLElement | null;

describe("MetricGauge", () => {
  it("shows three decimals and a green bar filled to the value", () => {
    const { container } = render(<MetricGauge value={0.998} />);
    expect(screen.getByText("0.998")).toBeInTheDocument();
    expect(bar(container)).toHaveAttribute("data-level", "good");
    expect((container.querySelector(".gauge__fill") as HTMLElement).style.transform).toBe("scaleX(0.998)");
  });

  it("turns the bar amber below the threshold, and green exactly at it", () => {
    const low = render(<MetricGauge value={METRIC_GOOD_MIN - 0.001} />);
    expect(bar(low.container)).toHaveAttribute("data-level", "low");
    const at = render(<MetricGauge value={METRIC_GOOD_MIN} />);
    expect(bar(at.container)).toHaveAttribute("data-level", "good");
  });

  it("hides the bar from assistive tech, because the number already says it", () => {
    const { container } = render(<MetricGauge value={0.5} />);
    expect(bar(container)).toHaveAttribute("aria-hidden", "true");
  });

  it("shows a muted n/a with no bar when the value is missing", () => {
    for (const v of [null, undefined, Number.NaN]) {
      const { container, unmount } = render(<MetricGauge value={v} />);
      expect(screen.getByText("n/a")).toHaveClass("gauge--na");
      expect(bar(container)).toBeNull();
      unmount();
    }
  });

  it("clamps out-of-range values for the bar but still prints the number", () => {
    const { container } = render(<MetricGauge value={1.2} />);
    expect(screen.getByText("1.200")).toBeInTheDocument();
    expect((container.querySelector(".gauge__fill") as HTMLElement).style.transform).toBe("scaleX(1)");
  });
});
