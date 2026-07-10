import { expect, type Page, test } from "@playwright/test";

/** (a) + (b) of B-QUAL4-SMOKE: the app boots outside any Tauri runtime
 * (a plain `bun run dev` tab, exactly like a developer opening
 * localhost:1420 in a browser by mistake — see the app's own banner
 * copy about this) and every workspace mounts via left-rail navigation
 * without throwing or logging a console error. No Tauri IPC is mocked
 * here; each workspace's empty state must render on its own with
 * `data` still `null` in the store.
 */

function trackConsoleErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") errors.push(msg.text());
  });
  page.on("pageerror", (err) => errors.push(err.message));
  return errors;
}

test("app boots and the shell renders", async ({ page }) => {
  const errors = trackConsoleErrors(page);

  await page.goto("/");

  await expect(page.getByRole("heading", { name: "calibration-rs" })).toBeVisible();
  // The index route redirects to /diagnose (router.tsx).
  await expect(page).toHaveURL(/#\/diagnose$/);
  // Empty-state copy, before any export is loaded.
  await expect(page.getByText(/Open an/)).toBeVisible();

  expect(errors).toEqual([]);
});

const WORKSPACES: { rail: string; hash: string; marker: RegExp }[] = [
  { rail: "Diagnose", hash: "#/diagnose", marker: /Open an/ },
  {
    rail: "3D viewer",
    hash: "#/viewer3d",
    marker: /Load a rig export to see cameras and target poses in 3D\./,
  },
  {
    rail: "Epipolar",
    hash: "#/epipolar",
    marker: /Load a rig export to inspect epipolar geometry between cameras\./,
  },
  {
    rail: "Depth (dense stereo)",
    hash: "#/depth",
    marker:
      /Load a rig export \(two cameras \+ extrinsics\) to compute dense disparity\./,
  },
  { rail: "Run calibration", hash: "#/run", marker: /Run calibration/ },
];

for (const ws of WORKSPACES) {
  test(`mounts the "${ws.rail}" workspace without console errors`, async ({ page }) => {
    const errors = trackConsoleErrors(page);

    await page.goto("/");
    await page.getByRole("link", { name: ws.rail }).click();

    await expect(page).toHaveURL(new RegExp(`${ws.hash}$`));
    await expect(page.getByText(ws.marker)).toBeVisible();

    expect(errors).toEqual([]);
  });
}
