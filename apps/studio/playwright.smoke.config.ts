import { defineConfig, devices } from "@playwright/test";

/** The bun-only smoke run: the plain Vite dev server with no back end
 * behind its `/api` proxy (it points at a port nothing listens on), so every
 * route must mount and show the connection error without throwing. No Rust
 * toolchain needed. The specs against the real server are
 * `playwright.config.ts`. */
export default defineConfig({
  testDir: "./e2e",
  testMatch: "smoke.spec.ts",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: [["list"]],
  use: {
    baseURL: "http://localhost:1420",
    trace: "on-first-retry",
  },
  webServer: {
    command: "bun run dev",
    url: "http://localhost:1420",
    env: { CALIBRATION_STUDIO_URL: "http://127.0.0.1:9" },
    // Every proxied request fails on purpose; its log is noise.
    stderr: "ignore",
    reuseExistingServer: false,
    timeout: 30_000,
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
