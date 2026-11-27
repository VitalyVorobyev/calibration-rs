import { expect, test, type Page } from "@playwright/test";
import { command, libraryRoot, openLoggedIn, trackErrors } from "./support";

/** The Calibrate screen through the real server: the parameter editor on the
 * fixture project `planar-demo` (six rendered chessboard views), a run with
 * one parameter changed, its place in the history with what it changed,
 * rerun from it, star and label, a cancelled run, a validation message, and a
 * failing run (`rig-demo`: two views, one short). */

const enc = encodeURIComponent;

async function openCalibrate(page: Page, dir: string) {
  await page.goto(`/#/calibrate?project=${enc(dir)}`);
  await expect(page.getByRole("heading", { name: "Parameters" })).toBeVisible();
}

async function setMaxIters(page: Page, value: number) {
  const field = page.getByRole("spinbutton", { name: /^Max iters/i });
  await field.fill(String(value));
  await field.blur();
}

test("edits a parameter, runs, finds the run with its change, reruns from it, stars and labels it", async ({
  page,
  request,
}) => {
  test.setTimeout(240_000);
  const errors = trackErrors(page);
  const root = await libraryRoot(request);
  const dir = `${root}/planar-demo`;
  await openLoggedIn(page);
  await openCalibrate(page, dir);

  // The editor at rest: no changes, the seed row says it is not there yet.
  await expect(page.getByText("No changes")).toBeVisible();
  await expect(page.getByText(/not available yet/)).toBeVisible();

  // One parameter changed: marked, counted, and listed against the project's.
  await setMaxIters(page, 30);
  await expect(page.getByText("1 change").first()).toBeVisible();
  const changed = page.getByRole("list", { name: "Changed fields" });
  await expect(changed).toContainText("max iters");
  await page.getByLabel("Label for this run").fill("thirty iterations");
  await page.getByRole("button", { name: "Run", exact: true }).click();

  // The run lists with its change, its RMS and its views; the project file
  // was not written by it.
  const runs = page.getByRole("list", { name: "Saved runs" });
  const card = runs.getByRole("button", { name: /^Run run-001/ });
  await expect(card).toBeVisible({ timeout: 120_000 });
  await expect(runs).toContainText("1 change");
  await expect(runs).toContainText("6/6 views");
  const listed = await command(request, "open_project", { dir });
  const spec = (listed.body as { spec: { config?: unknown } }).spec;
  expect(spec.config ?? {}).toEqual({});

  // Rerun from it: the editor takes its recorded settings.
  await page.getByRole("button", { name: "Actions for run-001" }).click();
  await page.getByRole("menuitem", { name: "Rerun from this run" }).click();
  await expect(page.getByText("Started from the settings of run-001")).toBeVisible();
  await expect(page.getByRole("spinbutton", { name: /^Max iters/i })).toHaveValue("30");

  // Star and label, saved to the run's notes.
  await page.getByRole("button", { name: "Star run-001" }).click();
  const label = runs.getByLabel("Run label");
  await label.fill("kept");
  await label.press("Enter");
  await expect(runs).toContainText("kept");
  await expect
    .poll(async () => {
      const r = await command(request, "list_runs", { dir });
      const first = (
        r.body as unknown as { notes: { label?: string; starred?: boolean } }[]
      )[0];
      return `${first?.notes.label}/${first?.notes.starred}`;
    })
    .toBe("kept/true");

  // Deleted, after the confirmation.
  await page.getByRole("button", { name: "Actions for run-001" }).click();
  await page.getByRole("menuitem", { name: "Delete…" }).click();
  await page.getByRole("dialog").getByRole("button", { name: "Delete" }).click();
  await expect(page.getByText(/No runs yet\. Set up/)).toBeVisible();
  expect(errors.filter((e) => !/422/.test(e))).toEqual([]);
});

test("says what the library says about a value, and will not run it", async ({
  page,
  request,
}) => {
  const root = await libraryRoot(request);
  await openLoggedIn(page);
  await openCalibrate(page, `${root}/planar-demo`);
  await setMaxIters(page, 0);
  await expect(page.getByText("solver.max_iters must be positive")).toBeVisible();
  await expect(page.getByRole("button", { name: "Run", exact: true })).toBeDisabled();
  await setMaxIters(page, 50);
  await expect(page.getByRole("button", { name: "Run", exact: true })).toBeEnabled();
});

test("cancels a run: no run is written", async ({ page, request }) => {
  test.setTimeout(120_000);
  const root = await libraryRoot(request);
  const dir = `${root}/planar-demo`;
  await openLoggedIn(page);
  await openCalibrate(page, dir);
  await page.getByText("Re-detect images").click();
  await page.getByRole("button", { name: "Run", exact: true }).click();
  const progress = page.getByRole("region", { name: "Calibration in progress" });
  await expect(progress).toBeVisible();
  await progress.getByRole("button", { name: "Cancel" }).click();
  await expect(page.getByText("Cancelled. No run was written.")).toBeVisible({
    timeout: 60_000,
  });
  const after = await command(request, "list_runs", { dir });
  expect(after.body).toEqual([]);
});

test("a failing run is said usefully and keeps the editor's settings", async ({
  page,
  request,
}) => {
  test.setTimeout(120_000);
  const root = await libraryRoot(request);
  await openLoggedIn(page);
  await openCalibrate(page, `${root}/rig-demo`);
  await page.getByRole("button", { name: "Run", exact: true }).click();
  await expect(page.getByText(/The calibration failed/)).toBeVisible({ timeout: 60_000 });
  await expect(page.getByTestId("failure-message")).toContainText("need 3, got 2");
  await expect(page.getByText(/too few usable views/)).toBeVisible();
  await expect(page.getByText(/still holds the settings that failed/)).toBeVisible();
});
