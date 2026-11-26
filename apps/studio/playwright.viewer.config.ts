import { defineConfig, devices } from "@playwright/test";

/** The browser viewer end to end: the built frontend (`bun run build`)
 * served by the real `calib-viewer` binary over two committed fixture runs
 * (`e2e/fixtures/`): a two-camera rig with images, as the app saves a run,
 * and the same export without an image list, as `vcal run -o` writes it.
 * Needs the Rust toolchain; `bun run test:e2e:viewer` builds the frontend
 * first. */
const viewer = (run: string, port: number) => ({
  command: `cargo run --quiet --manifest-path src-tauri/Cargo.toml --bin calib-viewer -- e2e/fixtures/${run} --port ${port} --no-open`,
  url: `http://127.0.0.1:${port}/api/session`,
  // The first run compiles the app crate.
  timeout: 900_000,
  reuseExistingServer: false,
});

export default defineConfig({
  testDir: "./e2e",
  testMatch: "viewer.spec.ts",
  forbidOnly: !!process.env.CI,
  reporter: [["list"]],
  use: { trace: "on-first-retry" },
  webServer: [viewer("viewer-run", 1431), viewer("viewer-run-vcal", 1432)],
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
