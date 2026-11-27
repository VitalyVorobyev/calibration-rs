import { expect, type Page, test } from "@playwright/test";

/** The bun-only smoke run: the app boots with no back end answering (a
 * plain `bun run dev` tab, its `/api` proxy pointing nowhere), says so, and
 * every workspace mounts via the rail without throwing or logging a console
 * error of its own. Each workspace's empty state must render on its own with
 * nothing loaded. The browser's own report of the failed `/api` requests is
 * the scenario, not an error of the app's. */

function trackConsoleErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error" && !msg.text().startsWith("Failed to load resource")) {
      errors.push(msg.text());
    }
  });
  page.on("pageerror", (err) => errors.push(err.message));
  return errors;
}

test("app boots, the shell renders and says there is no back end", async ({ page }) => {
  const errors = trackConsoleErrors(page);

  await page.goto("/");

  await expect(page.getByRole("heading", { name: "calibration-rs" })).toBeVisible();
  // The index route redirects to /diagnose (router.tsx).
  await expect(page).toHaveURL(/#\/diagnose$/);
  await expect(page.getByText("No back end answers")).toBeVisible();
  // Empty-state copy, before any export is loaded.
  await expect(page.getByText(/Open a project and choose/)).toBeVisible();

  expect(errors).toEqual([]);
});

const WORKSPACES: { rail: string; hash: string; marker: RegExp }[] = [
  { rail: "Diagnose", hash: "#/diagnose", marker: /Open a project and choose/ },
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
    await expect(page.getByText(ws.marker).first()).toBeVisible();
    await expect(page.getByText("No back end answers")).toBeVisible();

    expect(errors).toEqual([]);
  });
}
