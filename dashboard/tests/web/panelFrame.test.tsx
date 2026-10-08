import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { PanelFrame } from "../../web/src/components/PanelFrame";
import { StatusWord } from "../../web/src/components/StatusWord";

describe("PanelFrame", () => {
  it("loading: exposes Loading <title> to assistive tech, hides the skeleton, renders no children", () => {
    const { container } = render(<PanelFrame title="Issues" state="loading"><p>child content</p></PanelFrame>);
    expect(screen.getByText("Loading Issues")).toBeInTheDocument();
    expect(screen.queryByText("child content")).not.toBeInTheDocument();
    expect(container.querySelector(".skeleton")).toHaveAttribute("aria-hidden", "true");
    expect(screen.getByRole("region", { name: "Issues" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { level: 2, name: "Issues" })).toBeInTheDocument();
  });

  it("error: shows the message in an alert and calls onRetry from the Retry button", async () => {
    const onRetry = vi.fn();
    render(<PanelFrame title="Issues" state="error" error="GitHub CLI failed" onRetry={onRetry}><p>child content</p></PanelFrame>);
    expect(screen.getByRole("alert")).toHaveTextContent("GitHub CLI failed");
    expect(screen.queryByText("child content")).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(onRetry).toHaveBeenCalledTimes(1);
  });

  it("error without onRetry has no Retry button", () => {
    render(<PanelFrame title="Issues" state="error" error="nope" />);
    expect(screen.queryByRole("button", { name: "Retry" })).not.toBeInTheDocument();
  });

  it("ready: renders children, the count and the actions", () => {
    render(
      <PanelFrame title="Issues" count={7} state="ready" actions={<button>Refresh</button>}>
        <p>child content</p>
      </PanelFrame>,
    );
    expect(screen.getByText("child content")).toBeInTheDocument();
    expect(screen.getByText("7")).toHaveClass("mono");
    expect(screen.getByRole("button", { name: "Refresh" })).toBeInTheDocument();
    expect(screen.queryByText("Loading Issues")).not.toBeInTheDocument();
  });
});

describe("StatusWord", () => {
  it.each(["ok", "fail", "running", "idle", "attention"] as const)("%s shows an icon and its word, never colour alone", (kind) => {
    const { container } = render(<StatusWord kind={kind}>Word</StatusWord>);
    expect(screen.getByText("Word")).toBeInTheDocument();
    const svg = container.querySelector("svg");
    expect(svg).not.toBeNull();
    expect(svg).toHaveAttribute("aria-hidden", "true");
    expect(container.firstElementChild).toHaveAttribute("data-kind", kind);
  });
});
