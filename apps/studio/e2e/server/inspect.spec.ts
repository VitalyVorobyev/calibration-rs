import { expect, test, type Page } from "@playwright/test";
import { command, libraryRoot, openLoggedIn, trackErrors } from "./support";

/** The Inspect screen's per-camera tabs through the real server, on the
 * fixture project `planar-demo` (six rendered chessboard views, one camera):
 * a calibration run, then its parameters, the distortion and residual maps
 * with their readout and legend, the projection σ map, the residual records
 * linked to the frame viewer, and the quality checks. */

const enc = encodeURIComponent;

/** Make a run of the project through the Calibrate screen. */
async function calibrate(page: Page, dir: string) {
  await page.goto(`/#/calibrate?project=${enc(dir)}`);
  await expect(page.getByRole("heading", { name: "Parameters" })).toBeVisible();
  await page.getByRole("button", { name: "Run", exact: true }).click();
  const runs = page.getByRole("list", { name: "Saved runs" });
  await expect(runs.getByRole("button", { name: /^Run run-001/ })).toBeVisible({
    timeout: 120_000,
  });
}

async function openTab(page: Page, dir: string, tab: string) {
  await page.goto(`/#/inspect/${tab}?project=${enc(dir)}&run=run-001`);
}

test("Inspect reads a run: parameters, maps, residuals, quality", async ({
  page,
  request,
}) => {
  test.setTimeout(300_000);
  const errors = trackErrors(page);
  const dir = `${await libraryRoot(request)}/planar-demo`;
  await openLoggedIn(page);
  await calibrate(page, dir);
  try {
    // Parameters: every parameter of the camera with its value ± σ and unit.
    await openTab(page, dir, "parameters");
    await expect(page.getByTestId("model-summary")).toContainText("pinhole");
    await expect(page.getByTestId("value-fx")).toContainText("±");
    await expect(page.getByRole("button", { name: "Copy as JSON" })).toBeVisible();
    await expect(page.getByRole("region", { name: "Distortion" })).toBeVisible();

    // Distortion: the map over the sensor, its legend in pixels and a reading
    // under the pointer.
    await openTab(page, dir, "distortion");
    const stage = page.getByRole("application", { name: "Distortion map of camera 0" });
    await expect(stage).toBeVisible();
    await expect(page.getByText("The lens displaces a pixel by up to")).toBeVisible();
    await stage.hover({ position: { x: 300, y: 200 } });
    await expect(page.getByTestId("map-readout")).toContainText("|d|");
    await expect(page.getByTestId("map-readout")).toContainText("px");
    await page.getByRole("radio", { name: "dx" }).check({ force: true });
    await expect(page.getByText(/^dx$/).first()).toBeVisible();

    // Residuals: the map says where the systematic error peaks; a table row
    // selects its view and feature, and the frame viewer rings the feature.
    await openTab(page, dir, "residuals");
    await expect(
      page.getByRole("application", { name: "Residual map of camera 0" }),
    ).toBeVisible({ timeout: 30_000 });
    await expect(page.getByText(/The systematic error peaks at/)).toBeVisible();
    await expect(page.getByRole("img", { name: /Residuals of camera 0/ })).toBeVisible();
    const records = page.getByRole("table", { name: "Residual records" });
    await records.getByRole("row").nth(1).click();
    await page.getByRole("tab", { name: "Frames" }).click();
    await expect(page.getByTestId("selected-feature")).toBeAttached();
    await page.getByRole("tab", { name: "Residuals" }).click();
    await expect(
      page.getByRole("table", { name: "Reprojection statistics" }),
    ).toBeVisible();

    // Projection σ.
    await openTab(page, dir, "sigma");
    await expect(
      page.getByRole("application", { name: "Projection uncertainty of camera 0" }),
    ).toBeVisible();
    await expect(page.getByText(/is known to/)).toBeVisible();

    // Quality: the verdict in words and each check with its measured value.
    await openTab(page, dir, "quality");
    await expect(page.getByTestId("verdict")).toContainText(/Pass|Warning|Fail/);
    await expect(page.getByRole("list", { name: "Quality checks" })).toContainText(
      "limit",
    );
  } finally {
    // The fixture project is shared with the other specs: leave it as found.
    await command(request, "delete_run", { dir, id: "run-001" });
  }
  expect(errors).toEqual([]);
});

test("a rig run: choose the camera, then its parameters and maps", async ({
  page,
  request,
}) => {
  const errors = trackErrors(page);
  const dir = `${await libraryRoot(request)}/stereo-sn0042`;
  await openLoggedIn(page);
  await page.goto(`/#/inspect/parameters?project=${enc(dir)}&run=run-001`);

  // Camera 0, then camera 1: each its own parameters.
  await expect(page.getByTestId("model-summary")).toBeVisible();
  await page.getByRole("combobox", { name: "Camera" }).click();
  await page.getByRole("option", { name: "Camera 1" }).click();
  await expect(page.getByTestId("value-cam1.fx")).toContainText("±");
  await expect(page.getByTestId("value-cam0.fx")).toHaveCount(0);
  // Its pose in the rig, with a value: the baseline in millimetres.
  await expect(page.getByTestId("pose-cam1.cam_se3_rig")).toContainText(",");

  // The map tabs open for the camera chosen, a run without images included.
  await page.getByRole("tab", { name: "Residuals" }).click();
  await expect(
    page.getByRole("application", { name: "Residual map of camera 1" }),
  ).toBeVisible({ timeout: 30_000 });
  await expect(page.getByRole("switch", { name: "Image" })).toBeDisabled();
  await page.getByRole("tab", { name: "Distortion" }).click();
  await expect(
    page.getByRole("application", { name: "Distortion map of camera 1" }),
  ).toBeVisible();
  expect(errors).toEqual([]);
});

test("leaving the 3D scene for another screen logs no error", async ({
  page,
  request,
}) => {
  const errors = trackErrors(page);
  const root = await libraryRoot(request);
  await openLoggedIn(page);
  await page.goto(`/#/inspect/scene?project=${enc(`${root}/stereo-sn0042`)}`);
  await page.getByRole("button", { name: /^Run run-002/ }).click();
  // The active camera's label is the DOM the scene adds beside its canvas.
  await expect(page.getByText("cam 0", { exact: true }).first()).toBeVisible();
  for (const screen of ["Calibrate", "Library"]) {
    await page.getByRole("link", { name: screen }).first().click();
    await expect(page.getByRole("link", { name: screen }).first()).toHaveAttribute(
      "aria-current",
      "page",
    );
    if (screen === "Calibrate") {
      await page.getByRole("link", { name: "Inspect" }).click();
      await page.getByRole("tab", { name: "Scene" }).click();
      await expect(page.getByText("cam 0", { exact: true }).first()).toBeVisible();
    }
  }
  expect(errors).toEqual([]);
});
