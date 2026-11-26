import { defineConfig, devices } from "@playwright/test";

import { E2E_TOKEN } from "./e2e/server/support";

/** The screenshot suite (lab-ui PLAN L2): every workspace, empty and with the fixture
 * library's run open, against the real server (`scripts/e2e-server.mjs`, port 1433).
 *
 *     bun run test:screens                      # compare against the local baseline
 *     bun run test:screens --update-snapshots
 *
 * The baseline lives in `e2e/.screens/` and is not committed. It is captured on one
 * machine before a change and compared after it on the same machine: a toolchain upgrade
 * must not move a pixel. Kept apart from `playwright.config.ts` because a missing baseline
 * fails, and CI has none.
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
    baseURL: "http://127.0.0.1:1433",
  },
  webServer: {
    command: "bun scripts/e2e-server.mjs 1433",
    url: "http://127.0.0.1:1433/api/health",
    env: { STUDIO_E2E_TOKEN: E2E_TOKEN },
    reuseExistingServer: false,
    timeout: 900_000,
  },
});
