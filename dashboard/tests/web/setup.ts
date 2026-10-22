import "@testing-library/jest-dom/vitest";
import { afterEach } from "vitest";
import { cleanup } from "@testing-library/react";

// Vitest runs without globals, so Testing Library cannot register its own cleanup.
// Tab, filter and sort live in the query string, so each test starts from a clean address.
afterEach(() => {
  cleanup();
  window.history.replaceState(null, "", "/");
});
