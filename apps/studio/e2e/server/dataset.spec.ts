import { expect, test } from "@playwright/test";
import { command, imageWidth, libraryRoot, openLoggedIn, trackErrors } from "./support";

/** The Dataset screen on the real server and the fixture library: the
 * camera × view matrix with a missing image, the selected image, the draft
 * judged by the library as it is edited, Save over the revision, a file
 * changed on disk under the edits, and a manifest that is a file of its own. */

const enc = encodeURIComponent;

test("lays the dataset out as a camera by view matrix and shows the selected image", async ({
  page,
  request,
}) => {
  const errors = trackErrors(page);
  const root = await libraryRoot(request);
  await openLoggedIn(page);
  await page.goto(`/#/dataset?project=${enc(`${root}/rig-four-cam`)}`);

  const matrix = page.getByRole("grid", { name: "Camera by view" });
  await expect(matrix).toBeVisible();
  // Four cameras, eight views, the third camera has no image in view 4.
  for (const cam of ["cam0", "cam1", "cam2", "cam3"]) {
    await expect(matrix.getByRole("columnheader", { name: cam })).toBeVisible();
  }
  await expect(
    matrix.getByRole("gridcell", { name: "view 0004, camera cam2, no image" }),
  ).toBeVisible();

  // A cell shows its image, its path and its size on the right.
  await matrix.getByRole("gridcell", { name: /view 0002, camera cam1/ }).click();
  const detail = page.getByRole("complementary", { name: "Selected image" });
  await expect(detail).toContainText("images/cam1_0002.png");
  await expect(detail).toContainText("96 × 72 px");
  expect(await imageWidth(page, '[aria-label="Selected image"]')).toBe(96);
  // The same view in the other cameras, one click away.
  await detail.getByRole("button", { name: /cam3/ }).click();
  await expect(detail).toContainText("View 0002 · cam3");

  // Only the views with a problem.
  await page.getByRole("switch", { name: /Problems only/ }).click();
  await expect(matrix.getByRole("rowheader")).toHaveCount(1);
  await expect(matrix.getByRole("rowheader", { name: "0004" })).toBeVisible();
  expect(errors).toEqual([]);
});

test("edits the description, sees the library judge it, and saves", async ({
  page,
  request,
}) => {
  const root = await libraryRoot(request);
  const dir = `${root}/planar-demo`;
  await openLoggedIn(page);
  await page.goto(`/#/dataset?project=${enc(dir)}`);
  await expect(page.getByRole("grid", { name: "Camera by view" })).toBeVisible();
  const save = page.getByRole("button", { name: "Save", exact: true });
  await expect(save).toBeDisabled();

  // A target the library refuses: the error is under the field, the rail
  // counts it, and Save is refused.
  const rows = page.getByRole("spinbutton", { name: /^Rows/ }).first();
  await rows.fill("0");
  await expect(page.getByText(/invalid target/)).toBeVisible();
  // The rail counts the error on the Dataset link.
  await expect(page.getByRole("link", { name: /Dataset/ })).toContainText("1");
  await expect(save).toBeDisabled();
  await expect(
    page.getByRole("status").filter({ hasText: "Unsaved changes" }),
  ).toContainText("1 error");

  // Back to a valid one with another square size: Save writes it.
  await rows.fill("5");
  await page
    .getByRole("spinbutton", { name: /^Square size/ })
    .first()
    .fill("0.025");
  await expect(page.getByText(/invalid target/)).toHaveCount(0);
  await expect(save).toBeEnabled();
  await save.click();
  await expect(page.getByRole("status").filter({ hasText: /^Saved$/ })).toBeVisible();
  const { body } = await command(request, "open_project", { dir });
  const manifest = (
    body.spec as { dataset: { manifest: { target: { square_size_m: number } } } }
  ).dataset.manifest;
  expect(manifest.target.square_size_m).toBeCloseTo(0.025);
});

test("a camera is added with the next id with its pattern, a warning does not stop Save, and the draft is reverted", async ({
  page,
  request,
}) => {
  const root = await libraryRoot(request);
  await openLoggedIn(page);
  await page.goto(`/#/dataset?project=${enc(`${root}/rig-four-cam`)}`);
  await expect(page.getByRole("grid", { name: "Camera by view" })).toBeVisible();
  await page.getByRole("button", { name: "Duplicate cam1" }).click();
  // Its pattern followed the id; nothing matches it, which is a warning, not an error.
  await expect(page.getByRole("region", { name: "Camera cam4" })).toBeVisible();
  await expect(
    page.getByRole("textbox", { name: "Camera cam4 images pattern" }),
  ).toHaveValue("images/cam4_*.png");
  await expect(page.getByText("No image matches “images/cam4_*.png”.")).toBeVisible();
  await expect(page.getByRole("button", { name: "Save", exact: true })).toBeEnabled();
  await page.getByRole("button", { name: "Revert" }).click();
  await expect(page.getByRole("region", { name: "Camera cam4" })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Save", exact: true })).toBeDisabled();
});

test("a project file changed on disk under unsaved edits is said, and my edits go back on it", async ({
  page,
  request,
}) => {
  const root = await libraryRoot(request);
  const dir = `${root}/stereo-sn0042-field`;
  await openLoggedIn(page);
  await page.goto(`/#/dataset?project=${enc(dir)}`);
  await expect(page.getByRole("grid", { name: "Camera by view" })).toBeVisible();

  // An edit of mine: the target's rows.
  const rows = page.getByRole("spinbutton", { name: /^Rows/ }).first();
  await rows.fill("6");

  // Another client saves the project with a different square size.
  const opened = await command(request, "open_project", { dir });
  const spec = opened.body.spec as {
    dataset: { manifest: { target: { square_size_m: number } } };
  };
  spec.dataset.manifest.target.square_size_m = 0.03;
  const saved = await command(request, "save_project", {
    dir,
    revision: opened.body.revision,
    spec,
  });
  expect(saved.status).toBe(200);

  await expect(page.getByText("The project file changed on disk")).toBeVisible();
  // My edit is still there.
  await expect(rows).toHaveValue("6");
  await page.getByRole("button", { name: "Reload and keep my edits" }).click();
  await expect(page.getByText("The project file changed on disk")).toHaveCount(0);
  await expect(rows).toHaveValue("6");
  await expect(
    page.getByRole("spinbutton", { name: /^Square size/ }).first(),
  ).toHaveValue("0.03");
  await page.getByRole("button", { name: "Save", exact: true }).click();
  await expect(page.getByRole("status").filter({ hasText: /^Saved$/ })).toBeVisible();
  const after = await command(request, "open_project", { dir });
  const target = (
    after.body.spec as {
      dataset: { manifest: { target: { rows: number; square_size_m: number } } };
    }
  ).dataset.manifest.target;
  expect(target.rows).toBe(6);
  expect(target.square_size_m).toBeCloseTo(0.03);
});

test("a manifest that is a file of its own is shown read-only", async ({
  page,
  request,
}) => {
  const root = await libraryRoot(request);
  await openLoggedIn(page);
  // The project of the app's earlier format keeps its manifest in a file.
  await page.goto(`/#/dataset?project=${enc(`${root}/referenced-demo`)}`);
  await expect(
    page.getByText("Read-only: the manifest is a file of its own").first(),
  ).toBeVisible();
  await expect(page.getByRole("grid", { name: "Camera by view" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Save", exact: true })).toHaveCount(0);
  await expect(page.getByRole("textbox", { name: "Camera id" }).first()).toBeDisabled();
});
