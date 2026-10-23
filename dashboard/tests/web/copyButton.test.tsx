import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { CopyButton } from "../../web/src/components/CopyButton";

afterEach(() => {
  vi.restoreAllMocks();
  Reflect.deleteProperty(document, "execCommand");
});

describe("CopyButton", () => {
  it("says Copied when the clipboard accepts the text", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    vi.stubGlobal("navigator", { ...navigator, clipboard: { writeText } });
    render(<CopyButton text="abc1234" label="Copy commit hash" />);
    await userEvent.click(screen.getByRole("button", { name: "Copy commit hash" }));
    expect(writeText).toHaveBeenCalledWith("abc1234");
    expect(await screen.findByText("Copied")).toBeInTheDocument();
    vi.unstubAllGlobals();
  });

  it("falls back to execCommand when the clipboard API is unavailable", async () => {
    vi.stubGlobal("navigator", { ...navigator, clipboard: undefined });
    const exec = vi.fn().mockReturnValue(true);
    Object.defineProperty(document, "execCommand", { value: exec, configurable: true });
    render(<CopyButton text="abc1234" label="Copy commit hash" />);
    await userEvent.click(screen.getByRole("button", { name: "Copy commit hash" }));
    expect(exec).toHaveBeenCalledWith("copy");
    expect(await screen.findByText("Copied")).toBeInTheDocument();
    vi.unstubAllGlobals();
  });

  it("sets no timer when it unmounts while the copy is still pending", async () => {
    let finish: () => void = () => {};
    const writeText = vi.fn().mockReturnValue(new Promise<void>((r) => { finish = r; }));
    vi.stubGlobal("navigator", { ...navigator, clipboard: { writeText } });
    const timeout = vi.spyOn(globalThis, "setTimeout");
    const { unmount } = render(<CopyButton text="abc1234" label="Copy commit hash" />);
    await userEvent.click(screen.getByRole("button", { name: "Copy commit hash" }));
    const before = timeout.mock.calls.length;
    unmount();
    finish();
    await Promise.resolve();
    await Promise.resolve();
    expect(timeout.mock.calls.length).toBe(before);
    vi.unstubAllGlobals();
  });

  it("says Copy failed when nothing can copy", async () => {
    vi.stubGlobal("navigator", { ...navigator, clipboard: undefined });
    Object.defineProperty(document, "execCommand", { value: () => false, configurable: true });
    render(<CopyButton text="abc1234" label="Copy commit hash" />);
    await userEvent.click(screen.getByRole("button", { name: "Copy commit hash" }));
    expect(await screen.findByText("Copy failed")).toBeInTheDocument();
    vi.unstubAllGlobals();
  });
});
