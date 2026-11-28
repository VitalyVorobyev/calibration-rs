import { expect, test } from "@playwright/test";
import { command, libraryRoot, openLoggedIn, trackErrors } from "./support";

/** The Compare screen on the real server and the fixture library: the two
 * imported runs of `stereo-sn0042` (their relation is unknown, so every test
 * says why it is not made), the reference changed and the star kept, and two
 * real runs of `planar-demo` ticked in Calibrate's run list and compared (one
 * dataset: no test, a projection difference over the image). */

const enc = encodeURIComponent;

test.use({ viewport: { width: 1440, height: 900 } });

test("two imported runs: how they relate, every test with its reason, the map", async ({
  page,
  request,
}) => {
  const errors = trackErrors(page);
  const root = await libraryRoot(request);
  const dir = `${root}/stereo-sn0042`;
  await openLoggedIn(page);
  await page.goto(`/#/compare?project=${enc(dir)}&runs=run-001,run-002`);

  await expect(page.getByRole("heading", { name: "Compare" })).toBeVisible();
  await expect(page.getByRole("link", { name: "Compare" })).toHaveAttribute(
    "aria-current",
    "page",
  );
  // The relation is said before any number, and what it allows.
  const relation = page.getByRole("region", { name: "How the runs relate" });
  await expect(relation).toContainText("Relation unknown");
  await expect(relation).toContainText("no parameter test is made");

  // A test that is not made says why, never a statistic.
  const cam0 = page.getByRole("table", { name: "Parameters of cam0 across the runs" });
  const fx = cam0.getByRole("row", { name: /^fx/ });
  await expect(fx).toContainText("200.00 ± 0.50");
  await expect(fx).toContainText("200.60");
  await expect(fx).toContainText("not tested");
  await expect(fx).toContainText("relation unknown");
  await expect(fx).not.toContainText(/z [+−]?\d/);

  // Only the differences that matter: a change of 1 σ or more, here fx alone.
  await page.getByRole("switch", { name: "Only differences that matter" }).click();
  await expect(cam0.locator("tbody tr")).toHaveCount(1);
  await page.getByRole("switch", { name: "Only differences that matter" }).click();
  await expect(cam0.locator("tbody tr")).toHaveCount(4);

  // Fit quality side by side.
  await page.getByRole("tab", { name: "Fit quality" }).click();
  await expect(page).toHaveURL(/tab=fit/);
  await expect(page.getByRole("table", { name: "Fit quality of cam1" })).toBeVisible();

  // The projection difference map of the picked pair, loaded as a plane.
  await page.getByRole("tab", { name: "Projection difference" }).click();
  await expect(page.getByText(/run-002 projects within/)).toBeVisible();
  await expect(
    page.getByRole("application", { name: "Projection difference of cam0" }),
  ).toBeVisible();
  await page.getByRole("row", { name: /cam1/ }).click();
  // cam1's two calibrations project alike: nothing to draw, and it says so.
  await expect(
    page.getByText("Nothing to draw: the projections are the same"),
  ).toBeVisible();
  await page.getByRole("row", { name: /cam0/ }).click();
  await page.getByRole("radio", { name: "None" }).click();
  await expect(page.getByText(/frames as calibrated/)).toBeVisible();

  expect(errors).toEqual([]);
});

test("the reference changes the comparison, and a run is starred and unstarred from the screen", async ({
  page,
  request,
}) => {
  const errors = trackErrors(page);
  const root = await libraryRoot(request);
  const dir = `${root}/stereo-sn0042`;
  await openLoggedIn(page);
  await page.goto(`/#/compare?project=${enc(dir)}&runs=run-001,run-002`);
  await expect(page.getByText("2 runs · reference run-001")).toBeVisible();

  const second = page
    .getByRole("list", { name: "Runs compared" })
    .getByRole("listitem")
    .filter({ hasText: "run-002" });
  await second.getByRole("button", { name: "Make reference" }).click();
  await expect(page.getByText("2 runs · reference run-002")).toBeVisible();
  await expect(page).toHaveURL(/runs=run-002%2Crun-001|runs=run-002,run-001/);
  const cam0 = page.getByRole("table", { name: "Parameters of cam0 across the runs" });
  // The change is now run-001's, against run-002.
  await expect(cam0.getByRole("row", { name: /^fx/ })).toContainText("200.60");

  const starred = async (id: string) => {
    const r = await command(request, "list_runs", { dir });
    const runs = r.body as unknown as { id: string; notes: { starred?: boolean } }[];
    return runs.find((x) => x.id === id)?.notes.starred ?? false;
  };
  expect(await starred("run-001")).toBe(false);
  const tiles = page.getByRole("list", { name: "Runs compared" });
  await tiles.getByRole("button", { name: "Star run-001" }).click();
  await expect.poll(() => starred("run-001")).toBe(true);
  // The run list beside it shows the same star.
  await expect(
    page
      .getByRole("list", { name: "Saved runs" })
      .getByRole("button", { name: "Unstar run-001" }),
  ).toBeVisible();
  // Leave the fixture as it was.
  await tiles.getByRole("button", { name: "Unstar run-001" }).click();
  await expect.poll(() => starred("run-001")).toBe(false);
  expect(errors).toEqual([]);
});

test("two runs of one dataset, ticked in the run list: no test, and where the projections differ", async ({
  page,
  request,
}) => {
  test.setTimeout(240_000);
  const errors = trackErrors(page);
  const root = await libraryRoot(request);
  const dir = `${root}/planar-demo`;
  await openLoggedIn(page);
  await page.goto(`/#/calibrate?project=${enc(dir)}`);
  const run = page.getByRole("button", { name: "Run", exact: true });
  const saved = page.getByRole("list", { name: "Saved runs" });
  try {
    await run.click();
    await expect(saved.getByRole("button", { name: /^Run run-001/ })).toBeVisible({
      timeout: 120_000,
    });
    const iters = page.getByRole("spinbutton", { name: /^Max iters/i });
    await iters.fill("20");
    await iters.blur();
    await run.click();
    await expect(saved.getByRole("button", { name: /^Run run-002/ })).toBeVisible({
      timeout: 120_000,
    });

    await saved.getByRole("checkbox", { name: /run-001/ }).check();
    await saved.getByRole("checkbox", { name: /run-002/ }).check();
    await page.getByRole("button", { name: "Compare 2 ▸" }).click();

    await expect(page).toHaveURL(/#\/compare\?/);
    await expect(page.getByText("2 runs · reference run-001")).toBeVisible();
    const relation = page.getByRole("region", { name: "How the runs relate" });
    await expect(relation).toContainText("Same data");
    await expect(relation).toContainText("6 images and 210 features in common");
    await expect(relation).toContainText("solver.max_iters");
    await expect(page.getByText(/No parameter test applies to these runs/)).toBeVisible();
    const cam0 = page.getByRole("table", { name: "Parameters of cam0 across the runs" });
    await expect(cam0.getByRole("row", { name: /^fx/ })).toContainText("shared data");

    await page.getByRole("tab", { name: "Projection difference" }).click();
    // The same data under another iteration limit project every ray alike:
    // the screen says so instead of drawing rounding error.
    await expect(
      page.getByText("Nothing to draw: the projections are the same"),
    ).toBeVisible();
    // Before the runs are deleted: the open comparison would ask again.
    expect(errors.filter((e) => !/422/.test(e))).toEqual([]);
  } finally {
    for (const id of ["run-001", "run-002"]) {
      await command(request, "delete_run", { dir, id });
    }
  }
});
