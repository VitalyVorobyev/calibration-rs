import { expect, test, type APIRequestContext, type Page } from "@playwright/test";
import { command, imageWidth, libraryRoot, openLoggedIn, trackErrors } from "./support";

/** The Detect screen on the real server and the fixture library: the matrix
 * of counts filled by Detect all, the image with the detector's features
 * over it, picking and boxing features on the stage, every kind of
 * exclusion written to the project file (and undone), and detector
 * settings tried on one image before they are applied. */

const enc = encodeURIComponent;

/** The project's spec now, to put it back after a spec that writes. */
async function specOf(request: APIRequestContext, dir: string) {
  const { body } = await command(request, "open_project", { dir });
  return body.spec;
}

async function restore(request: APIRequestContext, dir: string, spec: unknown) {
  const { body } = await command(request, "open_project", { dir });
  await command(request, "save_project", { dir, revision: body.revision, spec });
}

async function excludeOf(request: APIRequestContext, dir: string) {
  const spec = (await specOf(request, dir)) as { exclude?: Record<string, unknown[]> };
  return spec.exclude ?? {};
}

/** Image point `(x, y)` of the stage's image as a point on the page. */
async function onStage(page: Page, x: number, y: number, size = { w: 320, h: 240 }) {
  const box = await page.locator("[data-stage]").first().boundingBox();
  if (!box) throw new Error("the stage has no box");
  return {
    x: box.x + ((x + 0.5) / size.w) * box.width,
    y: box.y + ((y + 0.5) / size.h) * box.height,
  };
}

/** Press Detect all (or Detect again) and wait for the job to end, so no
 * job of this spec is still running when another spec starts. */
async function detectAll(page: Page, name: string | RegExp = /^Detect (all|again)$/) {
  await page.getByRole("button", { name }).click();
  await expect(page.getByRole("button", { name: "Cancel" })).toHaveCount(0, {
    timeout: 60_000,
  });
}

async function openDetect(page: Page, root: string, project: string) {
  await openLoggedIn(page);
  await page.goto(`/#/detect?project=${enc(`${root}/${project}`)}`);
  await expect(
    page.getByRole("grid", { name: "Detections by camera and view" }),
  ).toBeVisible();
}

test("fills the matrix with Detect all and draws the detection over the image", async ({
  page,
  request,
}) => {
  const errors = trackErrors(page);
  const root = await libraryRoot(request);
  await openDetect(page, root, "rig-four-cam");
  const matrix = page.getByRole("grid", { name: "Detections by camera and view" });
  for (const cam of ["cam0", "cam1", "cam2", "cam3"]) {
    await expect(matrix.getByRole("columnheader", { name: cam })).toBeVisible();
  }
  // The third camera has no image in view 4.
  await expect(
    matrix.getByRole("gridcell", { name: /view 0004, camera cam2, –/ }),
  ).toBeVisible();

  // Detect all is a job; the counts fill in and a full board has all of its corners.
  await detectAll(page);
  await expect(
    matrix.getByRole("gridcell", {
      name: /view 0002, camera cam1, 20: 20 of 20 features/,
    }),
  ).toBeVisible({ timeout: 60_000 });
  await expect(page.getByText("Nothing has been detected")).toHaveCount(0);

  // The image, with its corners over it.
  await matrix.getByRole("gridcell", { name: /view 0002, camera cam1/ }).click();
  const stage = page.getByRole("application", { name: "Image images/cam1_0002.png" });
  await expect(stage).toBeVisible();
  expect(await imageWidth(page, '[aria-label="Image images/cam1_0002.png"]')).toBe(96);
  await expect(page.locator('[aria-label^="Corners: 20 points"]')).toHaveCount(1);
  await expect(page.getByRole("table", { name: "Features of this image" })).toBeVisible();
  await expect(page.getByText("20 of 20")).toBeVisible();

  // Stepping the camera and the view with the keys.
  await page.keyboard.press(".");
  await expect(page.getByRole("heading", { name: /View 0002/ })).toBeVisible();
  await expect(
    page.getByRole("application", { name: "Image images/cam2_0002.png" }),
  ).toBeVisible();
  await page.keyboard.press("]");
  await expect(
    page.getByRole("application", { name: "Image images/cam2_0003.png" }),
  ).toBeVisible();

  // A camera with no image says so.
  await matrix.getByRole("gridcell", { name: /view 0004, camera cam2/ }).click();
  await expect(page.getByText(/cam2 has no image for view 0004/)).toBeVisible();
  expect(errors).toEqual([]);
});

test("hovers a corner to read it, picks one on the stage and boxes the others", async ({
  page,
  request,
}) => {
  const root = await libraryRoot(request);
  const dir = `${root}/planar-demo`;
  const original = await specOf(request, dir);
  try {
    await openDetect(page, root, "planar-demo");
    await detectAll(page);
    const matrix = page.getByRole("grid", { name: "Detections by camera and view" });
    await expect(
      matrix.getByRole("gridcell", { name: /view 0, camera cam0, 35/ }),
    ).toBeVisible({
      timeout: 60_000,
    });
    await matrix.getByRole("gridcell", { name: /view 0, camera cam0/ }).click();
    await expect(page.locator('[aria-label^="Corners: 35 points"]')).toHaveCount(1);

    // The corner at (114.5, 89.5) is the board's first.
    const first = await onStage(page, 114.5, 89.5);
    await page.mouse.move(first.x, first.y);
    const region = page.getByRole("region", { name: "Image" });
    await expect(region.getByRole("list").filter({ hasText: "114.50" })).toBeVisible();

    // A press picks it; the overlay says one is selected.
    await page.mouse.click(first.x, first.y);
    await expect(
      page.locator('[aria-label^="Corners: 35 points, 1 selected"]'),
    ).toHaveCount(1);
    await expect(page.getByRole("button", { name: "Exclude 1 feature" })).toBeVisible();

    // ⇧-drag boxes the three corners of the board's first row.
    const a = await onStage(page, 100, 80);
    const b = await onStage(page, 152, 98);
    await page.mouse.move(a.x, a.y);
    await page.keyboard.down("Shift");
    await page.mouse.down();
    await page.mouse.move((a.x + b.x) / 2, (a.y + b.y) / 2, { steps: 4 });
    await page.mouse.move(b.x, b.y, { steps: 4 });
    await page.mouse.up();
    await page.keyboard.up("Shift");
    await expect(page.getByRole("button", { name: "Exclude 3 features" })).toBeVisible();

    // Excluding them writes their board points; they stay on the image, crossed.
    await page.getByRole("button", { name: "Exclude 3 features" }).click();
    await expect(page.locator('[aria-label^="Excluded features: 3 points"]')).toHaveCount(
      1,
    );
    await expect(page.getByRole("region", { name: "Exclusions" })).toContainText(
      "3 features",
    );
    const exclude = await excludeOf(request, dir);
    expect(exclude.features).toHaveLength(1);
    const entry = exclude.features?.[0] as {
      camera: string;
      image: string;
      points: number[][];
    };
    expect(entry).toMatchObject({ camera: "cam0", image: "images/view0.png" });
    expect(entry.points).toHaveLength(3);
    expect(entry.points[0]).toEqual([0, 0, 0]);
  } finally {
    await restore(request, dir, original);
  }
});

test("excludes an image and a view with the keys, marks them, undoes and restores", async ({
  page,
  request,
}) => {
  const root = await libraryRoot(request);
  const dir = `${root}/planar-demo`;
  const original = await specOf(request, dir);
  try {
    await openDetect(page, root, "planar-demo");
    const matrix = page.getByRole("grid", { name: "Detections by camera and view" });
    await matrix.getByRole("gridcell", { name: /view 1, camera cam0/ }).click();
    await expect(
      page.getByRole("application", { name: /Image images\/view1.png/ }),
    ).toBeVisible();

    await page.keyboard.press("x");
    await expect(
      matrix.getByRole("gridcell", { name: /view 1, camera cam0.*Excluded/ }),
    ).toBeVisible();
    await expect(page.getByText("Excluded from calibration").first()).toBeVisible();
    await expect(page.getByRole("button", { name: "Restore image" })).toBeVisible();
    expect(await excludeOf(request, dir)).toMatchObject({
      images: [{ camera: "cam0", image: "images/view1.png" }],
    });

    await page.keyboard.press("Shift+x");
    await expect(matrix.getByText("view excluded")).toBeVisible();
    expect(await excludeOf(request, dir)).toMatchObject({
      views: [{ view: { token: "1" } }],
    });

    // Undo takes the view back, then the image; the tray says what is left.
    await page.keyboard.press("Control+z");
    await expect(matrix.getByText("view excluded")).toHaveCount(0);
    const tray = page.getByRole("region", { name: "Exclusions" });
    await expect(tray).toContainText("1 image");
    await page.getByRole("button", { name: "Undo" }).click();
    await expect(tray).toContainText("Nothing excluded");
    expect(await excludeOf(request, dir)).toEqual({});

    // Only excluded: the filter finds them again after a toggle.
    await page.keyboard.press("x");
    await page.getByRole("switch", { name: "Only excluded" }).click();
    await expect(matrix.getByRole("rowheader")).toHaveCount(1);
    // The panel lists it and restores it.
    await page.getByRole("button", { name: /^Exclusions/ }).click();
    const dialog = page.getByRole("dialog");
    await expect(dialog.getByRole("row")).toHaveCount(2);
    await dialog.getByRole("button", { name: "Restore image images/view1.png" }).click();
    await expect(dialog.getByText(/Nothing is excluded/)).toBeVisible();
    expect(await excludeOf(request, dir)).toEqual({});
  } finally {
    await restore(request, dir, original);
  }
});

test("tries detector settings on one image, then applies them to the project", async ({
  page,
  request,
}) => {
  const root = await libraryRoot(request);
  const dir = `${root}/planar-demo`;
  const original = await specOf(request, dir);
  try {
    await openDetect(page, root, "planar-demo");
    await detectAll(page);
    const matrix = page.getByRole("grid", { name: "Detections by camera and view" });
    await expect(
      matrix.getByRole("gridcell", { name: /view 0, camera cam0, 35/ }),
    ).toBeVisible({
      timeout: 60_000,
    });
    await matrix.getByRole("gridcell", { name: /view 0, camera cam0/ }).click();

    await page.getByRole("tab", { name: "Detector" }).click();
    const apply = page.getByRole("button", { name: "Apply to project" });
    await expect(apply).toBeDisabled();
    // A corner must respond this strongly: no corner of the image does.
    await page.getByRole("switch", { name: /Chess corners/ }).click();
    const field = page.getByRole("spinbutton", { name: /Min corner strength/ });
    await field.fill("100000");
    await field.blur();
    await expect(page.getByText("changed").first()).toBeVisible();
    await page.getByRole("button", { name: "Preview on this image" }).click();
    await expect(page.getByText("Preview (unsaved settings)")).toBeVisible();
    await expect(page.getByText(/−35 features against the saved settings/)).toBeVisible();
    // Nothing was written, and the saved detection is one press away.
    expect(await specOf(request, dir)).toEqual(original);
    await page.getByRole("radio", { name: "Saved" }).click();
    await expect(page.locator('[aria-label^="Corners: 35 points"]')).toHaveCount(1);

    // Applying writes the manifest; the cache of the other images is the old settings'.
    await expect(apply).toBeEnabled();
    await apply.click();
    await expect(page.getByText("Detector settings saved")).toBeVisible();
    const saved = (await specOf(request, dir)) as {
      dataset: { manifest: { detector?: unknown } };
    };
    expect(saved.dataset.manifest.detector).toMatchObject({
      chess_corners: { min_corner_strength: 100000 },
    });
    // The other images hold the old settings' detections no more; the open
    // one was detected again with the new ones.
    await expect(
      matrix.getByRole("gridcell", { name: /view 1, camera cam0, ·/ }),
    ).toBeVisible();
    await expect(
      matrix.getByRole("gridcell", { name: /view 0, camera cam0, 0/ }),
    ).toBeVisible();

    // Detect all with the new settings: no corner is strong enough.
    await detectAll(page, "Detect all with the new settings");
    await expect(
      matrix.getByRole("gridcell", { name: /view 1, camera cam0, 0/ }),
    ).toBeVisible({
      timeout: 60_000,
    });
  } finally {
    await restore(request, dir, original);
  }
});

test("a manifest that is a file of its own previews but does not apply", async ({
  page,
  request,
}) => {
  const root = await libraryRoot(request);
  await openDetect(page, root, "referenced-demo");
  const matrix = page.getByRole("grid", { name: "Detections by camera and view" });
  await matrix.getByRole("gridcell").first().click();
  await page.getByRole("tab", { name: "Detector" }).click();
  await expect(
    page.getByText("Read-only: the manifest is a file of its own"),
  ).toBeVisible();
  await expect(page.getByRole("button", { name: "Apply to project" })).toBeDisabled();
  await expect(page.getByRole("button", { name: "Preview on this image" })).toBeEnabled();
});
