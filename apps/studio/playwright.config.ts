import { defineConfig, devices } from "@playwright/test";
import { E2E_PORT, E2E_TOKEN } from "./e2e/server/support";

/** Playwright against the real back end: the built frontend served by the
 * debug `calibration-studio-server` over a fresh copy of the committed
 * fixture library (`scripts/e2e-server.mjs`), on a loopback port with a
 * token. Needs the Rust toolchain (the script builds the server and the
 * frontend when they are missing). The bun-only smoke run is
 * `playwright.smoke.config.ts`. */
export default defineConfig({
  testDir: "./e2e/server",
  fullyParallel: false,
  workers: 1,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: [["list"]],
  use: {
    baseURL: `http://127.0.0.1:${E2E_PORT}`,
    trace: "on-first-retry",
  },
  webServer: {
    command: `bun scripts/e2e-server.mjs ${E2E_PORT}`,
    url: `http://127.0.0.1:${E2E_PORT}/api/health`,
    env: { STUDIO_E2E_TOKEN: E2E_TOKEN },
    // The first run may compile the server and build the frontend.
    timeout: 900_000,
    reuseExistingServer: false,
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
