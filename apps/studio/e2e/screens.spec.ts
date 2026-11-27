import { expect, test } from "@playwright/test";
import { E2E_TOKEN } from "./server/support";

/** One screenshot per screen address with nothing open, and Inspect with the
 * fixture library's run, all against the real server
 * (`scripts/e2e-server.mjs`). Run with `playwright.screens.config.ts`. */

const ADDRESSES = [
  "library",
  "calibrate",
  "inspect/frames",
  "inspect/scene",
  "inspect/epipolar",
  "inspect/depth",
  "dataset",
];

test.beforeEach(async ({ page }) => {
  await page.goto(`/#token=${E2E_TOKEN}`);
  await expect(page.getByRole("navigation", { name: "Screens" })).toBeVisible();
});

for (const address of ADDRESSES) {
  test(`${address}, empty`, async ({ page }) => {
    await page.goto(`/#/${address}`);
    await page.waitForLoadState("networkidle");
    await expect(page).toHaveScreenshot(`${address.replace("/", "-")}-empty.png`, {
      timeout: 30_000,
    });
  });
}

test("inspect, the fixture run", async ({ page }) => {
  await page.getByRole("button", { name: "Project", exact: true }).click();
  await page.getByRole("menuitem", { name: "Open project…" }).click();
  const picker = page.getByRole("dialog", { name: "Open a project" });
  await picker.getByRole("option", { name: /library$/ }).dblclick();
  await picker.getByRole("option", { name: /rig-demo/ }).click();
  await picker.getByRole("button", { name: "Choose", exact: true }).click();
  await page.getByRole("link", { name: "Inspect" }).click();
  await expect(page.getByRole("combobox", { name: "Run" })).toContainText("run-001");
  await expect(page.getByText("Rig extrinsics")).toBeVisible();
  await page.waitForLoadState("networkidle");
  await expect(page).toHaveScreenshot("inspect-rig.png", { timeout: 30_000 });
});
