import { expect, type Page, test } from "@playwright/test";

/** The browser viewer against its real HTTP back end (`calib-viewer`, see
 * `playwright.viewer.config.ts`): the page loads the run the server serves,
 * the viewer workspaces work, the desktop features are absent, and files
 * outside the run are refused. */

const WITH_IMAGES = "http://127.0.0.1:1431";
const WITHOUT_IMAGES = "http://127.0.0.1:1432";

function trackErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("pageerror", (err) => errors.push(err.message));
  page.on("console", (msg) => {
    if (msg.type() === "error") errors.push(msg.text());
  });
  return errors;
}

/** The decoded width of the `index`-th image under `scope`. */
async function imageWidth(page: Page, scope: string, index = 0): Promise<number> {
  const image = page.locator(`${scope} img`).nth(index);
  await expect(image).toBeAttached();
  await expect
    .poll(() => image.evaluate((img: HTMLImageElement) => img.naturalWidth))
    .toBeGreaterThan(0);
  return image.evaluate((img: HTMLImageElement) => img.naturalWidth);
}

test("opens the served run on Diagnose with its image, residuals and stats", async ({
  page,
}) => {
  const errors = trackErrors(page);
  await page.goto(`${WITH_IMAGES}/`);

  await expect(page).toHaveURL(/#\/diagnose$/);
  await expect(page.getByText("Rig extrinsics")).toBeVisible();
  await expect(page.getByText(/^mean reproj: \d+\.\d{3} px$/)).toBeVisible();
  // The run folder names what is loaded; no desktop controls.
  await expect(page.getByText("viewer-run", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Open Export…" })).toHaveCount(0);
  await expect(page.getByRole("link", { name: "Run calibration" })).toHaveCount(0);

  // The frame image arrives through the file route; the residual overlay
  // and its legend are drawn over it.
  expect(await imageWidth(page, '[aria-label="Frame viewer"]')).toBe(160);
  await expect(page.getByText("features 20")).toBeVisible();

  await page.getByRole("switch", { name: "Stats" }).click();
  await expect(page.getByLabel("Cross-camera residual matrix")).toBeVisible();
  await expect(page.getByLabel("Per-pose residual stats")).toBeVisible();

  await page.getByRole("switch", { name: "Uncertainty" }).click();
  await expect(page.getByLabel("Projection uncertainty map")).toBeVisible();

  expect(errors).toEqual([]);
});

test("renders the rig in 3D and its epipolar pair", async ({ page }) => {
  const errors = trackErrors(page);
  await page.goto(`${WITH_IMAGES}/`);
  await expect(page.getByText("Rig extrinsics")).toBeVisible();

  await page.getByRole("link", { name: "Epipolar" }).click();
  // Both panes show their camera's undistorted image (PNG from the server).
  expect(await imageWidth(page, "section", 0)).toBe(160);
  expect(await imageWidth(page, "section", 1)).toBe(160);
  await expect(page.getByText(/pane A · cam 0 · undistorted/)).toBeVisible();

  // Last: leaving the 3D viewer can log a React removeChild error from the
  // frustum labels' unmount, on either back end.
  await page.getByRole("link", { name: "3D viewer" }).click();
  await expect(page.locator("canvas").first()).toBeVisible();
  await expect(page.getByText("cameras 2 · poses 2")).toBeVisible();

  expect(errors).toEqual([]);
});

test("refuses files outside the run and commands it does not offer", async ({
  request,
}) => {
  const session = (await (await request.get(`${WITH_IMAGES}/api/session`)).json()) as {
    run_dir: string;
  };
  const inside = await request.get(
    `${WITH_IMAGES}/api/file?path=${encodeURIComponent(`${session.run_dir}/images/cam0_view0.png`)}`,
  );
  expect(inside.status()).toBe(200);
  const outside = await request.get(
    `${WITH_IMAGES}/api/file?path=${encodeURIComponent(`${session.run_dir}/../viewer-run-vcal/calibration.json`)}`,
  );
  expect(outside.status()).toBe(403);
  const run = await request.post(`${WITH_IMAGES}/api/cmd/run_calibration`, { data: {} });
  expect(run.status()).toBe(404);
});

test("opens an export without an image list", async ({ page }) => {
  const errors = trackErrors(page);
  await page.goto(`${WITHOUT_IMAGES}/`);
  await expect(page.getByText("Rig extrinsics")).toBeVisible();
  await expect(page.getByText(/This export has no image list/)).toBeVisible();
  await expect(page.getByText("features 20")).toBeVisible();

  await page.getByRole("switch", { name: "Stats" }).click();
  await expect(page.getByLabel("Cross-camera residual matrix")).toBeVisible();

  await page.getByRole("link", { name: "Epipolar" }).click();
  await expect(page.getByText(/This export has no image list/).first()).toBeVisible();

  await page.getByRole("link", { name: "3D viewer" }).click();
  await expect(page.locator("canvas").first()).toBeVisible();

  expect(errors).toEqual([]);
});
