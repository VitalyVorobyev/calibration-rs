import { expect, test } from "@playwright/test";
import { installTauriMock } from "./support/tauriMock";
import { ONE_PX_PNG_DATA_URL, PLANAR_EXPORT_FIXTURE } from "./support/fixtures";

/** One screenshot per workspace with nothing loaded, and Diagnose with the planar fixture
 * loaded through the mocked Tauri IPC (the same path as `diagnose.spec.ts`). Run with
 * `playwright.screens.config.ts`. */

const WORKSPACES = ["diagnose", "viewer3d", "epipolar", "depth", "run"];

for (const ws of WORKSPACES) {
  test(`${ws}, empty`, async ({ page }) => {
    await page.goto(`/#/${ws}`);
    await page.waitForLoadState("networkidle");
    await expect(page).toHaveScreenshot(`${ws}-empty.png`, { timeout: 30_000 });
  });
}

test("diagnose, planar export", async ({ page }) => {
  await installTauriMock(page, {
    "plugin:dialog|open": "/fake/export.json",
    load_export: { export: PLANAR_EXPORT_FIXTURE, export_dir: "/fake" },
    set_active_export: null,
    load_image: ONE_PX_PNG_DATA_URL,
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Open Export…" }).click();
  await expect(page.getByText("mean reproj: 0.220 px")).toBeVisible();
  await page.waitForLoadState("networkidle");
  await expect(page).toHaveScreenshot("diagnose-planar.png", { timeout: 30_000 });
});
