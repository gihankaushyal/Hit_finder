import { describe, it, expect, afterEach } from "vitest";
import { render, screen, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ToastHost } from "../../web/src/components/ToastHost";
import { clearToasts, pushToast } from "../../web/src/lib/attentionEffects";

afterEach(() => clearToasts());

describe("ToastHost", () => {
  it("lists toasts and lets the user dismiss one", async () => {
    render(<ToastHost />);
    act(() => pushToast("a failed", "red"));
    expect(screen.getByText("a failed")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Dismiss: a failed" }));
    expect(screen.queryByText("a failed")).toBeNull();
  });

  it("is not a live region (the nav bubble already announces changes)", () => {
    render(<ToastHost />);
    act(() => pushToast("x", "green"));
    expect(screen.queryByRole("status")).toBeNull();
    expect(screen.queryByRole("alert")).toBeNull();
  });
});
