import { expect, test } from "@playwright/test";
import { E2E_TOKEN } from "./server/support";

/** One screenshot per workspace with nothing loaded, and Diagnose with the
 * fixture library's run open, all against the real server
 * (`scripts/e2e-server.mjs`). Run with `playwright.screens.config.ts`. */

const WORKSPACES = ["diagnose", "viewer3d", "epipolar", "depth", "run"];

test.beforeEach(async ({ page }) => {
  await page.goto(`/#token=${E2E_TOKEN}`);
  await expect(page.getByRole("heading", { name: "calibration-rs" })).toBeVisible();
});

for (const ws of WORKSPACES) {
  test(`${ws}, empty`, async ({ page }) => {
    await page.goto(`/#/${ws}`);
    await page.waitForLoadState("networkidle");
    await expect(page).toHaveScreenshot(`${ws}-empty.png`, { timeout: 30_000 });
  });
}

test("diagnose, the fixture run", async ({ page }) => {
  await page.getByRole("button", { name: /Project/ }).click();
  await page.getByRole("menuitem", { name: "Open project…" }).click();
  const picker = page.getByRole("dialog", { name: "Open a project" });
  await picker.getByRole("option").first().dblclick();
  await picker.getByRole("option", { name: /rig-demo/ }).click();
  await picker.getByRole("button", { name: "Choose", exact: true }).click();
  await page.getByRole("tree", { name: "Saved runs" }).getByText("run-001").click();
  await expect(page.getByText("Rig extrinsics")).toBeVisible();
  await page.waitForLoadState("networkidle");
  await expect(page).toHaveScreenshot("diagnose-rig.png", { timeout: 30_000 });
});
