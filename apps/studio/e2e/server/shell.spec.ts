import { expect, test } from "@playwright/test";
import { command, libraryRoot, openLoggedIn, trackErrors } from "./support";

/** The app's frame against the real server: the rail and its disabled
 * screens, a deep link (project, run and tab in the address), the standing
 * event stream (another client's change reaches this page), the job tray with
 * its Cancel, the keyboard, and a dropped connection that comes back. */

const enc = encodeURIComponent;

test("the rail leads to the screens, and a screen that is not built is disabled", async ({
  page,
}) => {
  const errors = trackErrors(page);
  await openLoggedIn(page);
  const rail = page.getByRole("navigation", { name: "Screens" });
  await expect(rail.getByRole("link")).toHaveText([
    "Library",
    "Dataset",
    "Detect",
    "Calibrate",
    "Inspect",
    "Compare",
    "Device",
  ]);
  // No project: the screens that read one wait for it; the rest are not built.
  for (const name of ["Dataset", "Detect", "Calibrate", "Inspect", "Compare", "Device"]) {
    await expect(rail.getByRole("link", { name })).toHaveAttribute(
      "aria-disabled",
      "true",
    );
  }
  await expect(rail.getByRole("link", { name: "Library" })).toHaveAttribute(
    "aria-current",
    "page",
  );
  // An address of an unbuilt screen (Compare) says so instead of breaking.
  await page.goto("/#/compare");
  await expect(page.getByText("Coming in a later release.")).toBeVisible();
  expect(errors).toEqual([]);
});

test("a link opens the project, the run and the tab it names, and the address follows", async ({
  page,
  request,
}) => {
  const errors = trackErrors(page);
  const root = await libraryRoot(request);
  await openLoggedIn(page);
  await page.goto(`/#/inspect/epipolar?project=${enc(`${root}/rig-demo`)}&run=run-001`);

  await expect(page.getByRole("button", { name: "Project", exact: true })).toContainText(
    "rig-demo",
  );
  await expect(page.getByRole("combobox", { name: "Run" })).toContainText("run-001");
  await expect(page.getByRole("tab", { name: "Epipolar" })).toHaveAttribute(
    "aria-selected",
    "true",
  );
  // Another screen keeps the project and run in its address; back returns.
  await page.getByRole("link", { name: "Calibrate" }).click();
  await expect(page).toHaveURL(/#\/calibrate\?project=.*&run=run-001$/);
  await page.goBack();
  await expect(page).toHaveURL(/#\/inspect\/epipolar\?project=/);
  await expect(page.getByRole("tab", { name: "Epipolar" })).toHaveAttribute(
    "aria-selected",
    "true",
  );
  expect(errors).toEqual([]);
});

test("a project opened without a run opens on its newest, and the address says which", async ({
  page,
  request,
}) => {
  const root = await libraryRoot(request);
  await openLoggedIn(page);
  await page.goto(`/#/inspect/frames?project=${enc(`${root}/rig-demo`)}`);
  await expect(page.getByRole("combobox", { name: "Run" })).toContainText("run-001");
  await expect(page).toHaveURL(/run=run-001$/);
});

test("another client's change reaches this page through the standing event stream", async ({
  page,
  request,
}) => {
  const root = await libraryRoot(request);
  await openLoggedIn(page);
  await page.goto(`/#/inspect/frames?project=${enc(`${root}/rig-demo`)}&run=run-001`);
  const runs = page.getByRole("list", { name: "Saved runs" });
  await expect(runs.getByText("run-001", { exact: true })).toBeVisible();

  // A second client labels the run: this page was not asked to reload.
  await command(request, "set_run_notes", {
    dir: `${root}/rig-demo`,
    id: "run-001",
    notes: { label: "from elsewhere" },
  });
  await expect(runs.getByText("from elsewhere")).toBeVisible({
    timeout: 10_000,
  });
});

test("the job tray follows a queue and cancels a queued job", async ({
  page,
  request,
}) => {
  test.setTimeout(150_000);
  const errors = trackErrors(page);
  const root = await libraryRoot(request);
  await openLoggedIn(page);
  await page.goto(`/#/calibrate?project=${enc(`${root}/planar-demo`)}`);
  await expect(page.getByRole("button", { name: "Run", exact: true })).toBeVisible();

  // Six runs of the project, one at a time: the last waits behind five.
  const ids: string[] = [];
  for (let i = 0; i < 6; i++) {
    const started = await command(request, "calibrate", {
      dir: `${root}/planar-demo`,
      redetect: true,
    });
    ids.push(started.body.id as string);
  }
  await page.getByRole("button", { name: /^Jobs/ }).click();
  const tray = page.getByRole("region", { name: "Jobs" });
  await expect(tray).toBeVisible();
  const last = ids[5];
  const row = tray.locator("li", { hasText: last });
  await expect(row).toContainText("Queued");
  await row.getByRole("button", { name: "Cancel" }).click();
  await expect(row).toContainText("Cancelled", { timeout: 15_000 });

  // The rest run to their end; the tray keeps the last three that ended (two
  // done, and the cancelled one).
  await expect(tray.getByText("Done")).toHaveCount(2, { timeout: 100_000 });
  expect(errors).toEqual([]);
});

test("the keyboard: the palette goes to a screen, ? lists the shortcuts", async ({
  page,
  request,
}) => {
  const root = await libraryRoot(request);
  await openLoggedIn(page);
  await page.goto(`/#/library?project=${enc(`${root}/rig-demo`)}`);
  await expect(page.getByRole("button", { name: "Project", exact: true })).toContainText(
    "rig-demo",
  );

  await page.keyboard.press("Control+k");
  const input = page.getByRole("combobox", { name: "Search commands" });
  await expect(input).toBeFocused();
  await input.fill("calib");
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(/#\/calibrate\?project=/);

  await page.keyboard.press("Control+5");
  await expect(page).toHaveURL(/#\/inspect\/frames\?project=/);

  await page.keyboard.press("?");
  await expect(page.getByRole("dialog", { name: "Keyboard shortcuts" })).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).toHaveCount(0);
});

test("a dropped event stream says so, and clears when it is back", async ({ page }) => {
  await page.route("**/api/events**", (route) => route.abort());
  await openLoggedIn(page);
  await expect(page.getByText(/Connection lost/)).toBeVisible();
  await expect(page.getByText("reconnecting", { exact: true })).toBeVisible();
  await page.unroute("**/api/events**");
  // The browser retries the stream by itself.
  await expect(page.getByText(/Connection lost/)).toHaveCount(0, { timeout: 30_000 });
});
