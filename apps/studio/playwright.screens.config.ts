import { defineConfig, devices } from "@playwright/test";

/** The screenshot suite (lab-ui PLAN L2): every workspace, empty and with the planar
 * fixture loaded through the mocked Tauri IPC.
 *
 *     bun run test:screens                      # compare against the local baseline
 *     bun run test:screens --update-snapshots
 *
 * The baseline lives in `e2e/.screens/` and is not committed. It is captured on one
 * machine before a change and compared after it on the same machine: a toolchain upgrade
 * must not move a pixel. Kept apart from `playwright.config.ts` because a missing baseline
 * fails, and CI's smoke run has none.
 */
export default defineConfig({
  testDir: "./e2e",
  testMatch: "screens.spec.ts",
  snapshotPathTemplate: "{testDir}/.screens/{arg}{ext}",
  fullyParallel: false,
  workers: 1,
  reporter: [["list"]],
  expect: { toHaveScreenshot: { maxDiffPixelRatio: 0.001, animations: "disabled" } },
  use: {
    ...devices["Desktop Chrome"],
    viewport: { width: 1440, height: 900 },
    baseURL: "http://localhost:1421",
  },
  webServer: {
    // Its own port, so it neither needs nor collides with a running `bun run dev`.
    command: "bunx vite --port 1421 --strictPort",
    url: "http://localhost:1421",
    reuseExistingServer: false,
    timeout: 30_000,
  },
});
