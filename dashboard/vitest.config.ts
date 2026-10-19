import { defineConfig } from "vitest/config";

// environmentMatchGlobs was removed in Vitest 5; projects give the same split.
export default defineConfig({
  test: {
    projects: [
      {
        test: {
          name: "server",
          environment: "node",
          include: ["tests/**/*.test.{ts,tsx}"],
          exclude: ["tests/web/**", "node_modules/**"],
        },
      },
      {
        test: {
          name: "web",
          environment: "jsdom",
          include: ["tests/web/**/*.test.{ts,tsx}"],
          setupFiles: ["tests/web/setup.ts"],
        },
      },
    ],
  },
});
