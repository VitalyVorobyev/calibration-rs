import { defineConfig, devices } from "@playwright/test";

/** Playwright smoke config (B-QUAL4). Boots the plain Vite dev server
 * (`bun run dev`, not `bun run tauri dev` — Tauri isn't available
 * headlessly in CI) on the same port the app always uses
 * (`vite.config.ts`'s `server.port: 1420`), then drives it in a real
 * Chromium tab. The app runs outside a Tauri webview here, so
 * `isTauriContext()` is false unless a spec injects
 * `window.__TAURI_INTERNALS__` itself (see `e2e/support/tauriMock.ts`) —
 * that's the Tauri-native-vs-mocked-IPC boundary documented in
 * `app/README.md`.
 */
export default defineConfig({
  testDir: "./e2e",
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
    reuseExistingServer: !process.env.CI,
    timeout: 30_000,
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
