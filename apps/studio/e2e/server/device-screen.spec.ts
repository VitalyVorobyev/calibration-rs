import { expect, test } from "@playwright/test";
import type { ProjectView } from "../../src/api/generated/types";
import { command, libraryRoot, openLoggedIn, trackErrors } from "./support";

/** The Device screen on the real server and the fixture library: the
 * history of `Stereo-X1 / SN-0042` across its two projects, a trend chart
 * with the library's spread and its linked selection, the units of the
 * type, and a seed previewed, written to a file (asking before it replaces
 * one) and set on, then cleared from, the open project. */

const enc = encodeURIComponent;
const DEVICE = "/#/device?product=Stereo-X1&device=SN-0042";

test.use({ viewport: { width: 1500, height: 950 } });

test("shows a device's runs across its projects, the representative one named", async ({
  page,
}) => {
  const errors = trackErrors(page);
  await openLoggedIn(page);
  await page.goto(DEVICE);
  const table = page.getByRole("table", { name: "Runs of the device" });
  await expect(table.locator("tbody tr")).toHaveCount(3);
  const bench2 = table.getByRole("row", { name: /run-002.*Bench calibration/ });
  await expect(bench2).toContainText("representative");
  await expect(bench2).toContainText("0.231");
  await expect(bench2).toContainText("0.247");
  await expect(table.getByRole("row", { name: /Field recheck/ })).toContainText(
    "run-001",
  );
  await expect(page.getByRole("link", { name: "Device" })).toHaveAttribute(
    "aria-current",
    "page",
  );

  // Starred runs only is in the address.
  await page.getByRole("switch", { name: "Starred runs only" }).click();
  await expect(table.locator("tbody tr")).toHaveCount(1);
  await expect(page).toHaveURL(/starred=1/);

  // A run opens in Inspect.
  await page.getByRole("button", { name: /Open run-002 of Bench calibration/ }).click();
  await expect(page).toHaveURL(/#\/inspect\/frames\?project=.*stereo-sn0042&run=run-002/);
  expect(errors).toEqual([]);
});

test("draws a parameter over the runs with its spread, and links a hovered run to the list", async ({
  page,
}) => {
  const errors = trackErrors(page);
  await openLoggedIn(page);
  await page.goto(`${DEVICE}&tab=trends`);
  const chart = page.getByRole("img", { name: "cam0.fx over the runs" });
  await expect(chart).toBeVisible();
  // The library's spread of the three runs' values (200, 200.6, 199.8).
  await expect(
    page.getByText(/median 200 · robust σ 0\.297 · min 199\.8 · max 200\.6/).first(),
  ).toBeVisible();
  await expect(page.getByText("Ticks on the axis:")).toBeVisible();

  // Hovering the chart marks the nearest run in the list beside it.
  const box = (await chart.boundingBox())!;
  await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.45);
  await expect(page.locator('tr[aria-current="true"]')).toHaveCount(1);
  await page.mouse.move(box.x + box.width * 0.2, box.y + box.height * 0.45);
  await expect(page.locator('tr[aria-current="true"]')).toContainText(/run-001/);

  // The groups are tabs; a pose's σ is said to be of a perturbation.
  await page.getByRole("tab", { name: "Rig poses" }).click();
  await expect(page.getByText("σ of a perturbation").first()).toBeVisible();
  // The x axis can be the recorded time.
  await page.getByRole("radio", { name: "Recorded time" }).click();
  await expect(page).toHaveURL(/x=time/);
  expect(errors).toEqual([]);
});

test("compares the units of a type, and opens one", async ({ page }) => {
  await openLoggedIn(page);
  await page.goto("/#/device?product=Stereo-X1&scope=type");
  const table = page.getByRole("table", { name: "Units of the device type" });
  await expect(table.getByRole("row", { name: /Bench cell 3/ })).toContainText("run-002");
  await expect(table.getByRole("row", { name: /SN-0057/ })).toContainText("none");
  await page.getByRole("tab", { name: "Trends" }).click();
  await expect(page.getByRole("img", { name: "cam0.fx over the runs" })).toBeVisible();
  await page.getByRole("tab", { name: "Units" }).click();
  await page.getByRole("button", { name: "Show the history of SN-0057" }).click();
  await expect(page).toHaveURL(/device=SN-0057/);
  await expect(page.getByRole("table", { name: "Runs of the device" })).toBeVisible();
});

test("a Library device opens its history, and a project with no device explains", async ({
  page,
  request,
}) => {
  const root = await libraryRoot(request);
  await openLoggedIn(page);
  await page.goto("/#/library");
  await page.getByRole("treeitem", { name: /Bench cell 3 · SN-0042/ }).click();
  await page.getByRole("link", { name: "Device history" }).click();
  await expect(page).toHaveURL(/#\/device\?.*device=SN-0042/);
  await expect(page.getByRole("table", { name: "Runs of the device" })).toBeVisible();

  // A project that names no device: how to set one.
  await page.goto(`/#/device?project=${enc(`${root}/planar-demo`)}`);
  await expect(page.getByText(/names no device/)).toBeVisible();
  await page.getByRole("button", { name: "Set device…" }).click();
  await expect(page.getByRole("dialog", { name: "Set device" })).toBeVisible();
});

test("previews a seed, writes it to a file, and sets it on the open project and clears it", async ({
  page,
  request,
}) => {
  const errors = trackErrors(page);
  const root = await libraryRoot(request);
  const project = `${root}/stereo-sn0042-field`;
  // The fixture runs are undistorted: the project's camera model must be too
  // for the library to take their seed (put back at the end).
  const opened = (await command(request, "open_project", { dir: project }))
    .body as unknown as ProjectView;
  const undistorted = await command(request, "save_project", {
    dir: project,
    revision: opened.revision,
    spec: { ...opened.spec, config: { camera: { model: { distortion: "none" } } } },
  });
  expect(undistorted.status).toBe(200);
  await openLoggedIn(page);
  await page.goto(`${DEVICE}&tab=seed&project=${enc(project)}`);

  await expect(page.getByText("A seed is a starting value")).toBeVisible();
  await expect(
    page.getByText(/still estimates its values from its own images/),
  ).toBeVisible();
  const values = page.getByRole("table", { name: "Values of the seed" });
  await expect(values.getByRole("row", { name: /cam0\.fx/ })).toContainText("200.6");

  // Chosen runs: an aggregate, with each value's spread.
  await page.getByRole("radio", { name: "Chosen runs" }).click();
  await expect(page.getByText(/at least two runs/)).toBeVisible();
  await page.getByRole("checkbox", { name: /Bench calibration · run-002/ }).click();
  await page.getByRole("checkbox", { name: /Field recheck · run-001/ }).click();
  const spread = page.getByRole("table", {
    name: "Values of the seed and their spread across the runs",
  });
  await expect(spread.getByRole("row", { name: /cam0\.fx/ })).toContainText("199.8");

  // Written to a file chosen in the in-app picker; not over an existing one unasked.
  await page.getByRole("button", { name: "Choose…" }).click();
  const picker = page.getByRole("dialog", {
    name: "Choose the folder to write the seed to",
  });
  await picker.getByRole("option", { name: root }).dblclick();
  await picker.getByRole("option", { name: /^stereo-sn0057/ }).click();
  await picker.getByRole("button", { name: "Choose", exact: true }).click();
  await page.getByLabel("File name").fill("ui-seed.toml");
  await page.getByRole("button", { name: "Export", exact: true }).click();
  await expect(page.getByText("Seed written")).toBeVisible();
  await expect(page.getByText(/stereo-sn0057\/ui-seed\.toml/)).toBeVisible();

  await page.getByRole("button", { name: "Export", exact: true }).click();
  const confirm = page.getByRole("dialog", { name: "Replace the file?" });
  await expect(confirm).toBeVisible();
  await confirm.getByRole("button", { name: "Cancel" }).click();
  await expect(confirm).toHaveCount(0);

  // The open project (the field recheck) takes one run's seed, and clears it.
  await page.getByRole("radio", { name: "This run" }).click();
  await expect(page.getByText(/starts its calibrations from/)).toContainText("none");
  await page.getByRole("button", { name: "Use this run as the project's seed" }).click();
  await expect(page.getByText(/starts its calibrations from/)).toContainText(
    "stereo-sn0042/runs/run-002",
  );
  await page.getByRole("button", { name: "Clear seed" }).click();
  await expect(page.getByText(/starts its calibrations from/)).toContainText("none");
  // The 409 of the file that exists is the question asked, not an error of the page.
  expect(errors.filter((e) => !e.includes("409 (Conflict)"))).toEqual([]);

  const now = (await command(request, "open_project", { dir: project }))
    .body as unknown as ProjectView;
  const restored = await command(request, "save_project", {
    dir: project,
    revision: now.revision,
    spec: opened.spec,
  });
  expect(restored.status).toBe(200);
});

test("the back end's refusal of a seed that does not fit is shown, and nothing is written", async ({
  page,
  request,
}) => {
  const root = await libraryRoot(request);
  // The second unit's project has a distorting camera model; its runs' seed has none.
  const project = `${root}/stereo-sn0057`;
  await openLoggedIn(page);
  await page.goto(
    `/#/device?product=Stereo-X1&device=SN-0057&tab=seed&project=${enc(project)}`,
  );
  await expect(page.getByRole("table", { name: "Values of the seed" })).toBeVisible();
  await page.getByRole("button", { name: "Use this run as the project's seed" }).click();
  await expect(page.getByText("The project cannot take this seed")).toBeVisible();
  await expect(page.getByText(/the seed's model is/)).toBeVisible();
  await expect(page.getByText(/starts its calibrations from/)).toContainText("none");
  const view = await command(request, "open_project", { dir: project });
  expect((view.body as { spec: { seed?: unknown } }).spec.seed).toBeUndefined();
});

test("the breadcrumb's device lists open the Device screen, and Calibrate's seed row sets and clears the project's seed", async ({
  page,
  request,
}) => {
  const errors = trackErrors(page);
  const root = await libraryRoot(request);
  const project = `${root}/stereo-sn0042`;
  await openLoggedIn(page);
  await page.goto(`/#/calibrate?project=${enc(project)}`);
  await expect(page.getByRole("heading", { name: "Parameters" })).toBeVisible();

  // The device list goes to the Device screen on the chosen unit.
  await page.getByRole("button", { name: /^Device: choose another/ }).click();
  await page.getByRole("button", { name: /SN-0057/ }).click();
  await expect(page).toHaveURL(/#\/device\?.*device=SN-0057/);
  await expect(page.getByRole("table", { name: "Runs of the device" })).toBeVisible();

  // The type list shows every unit of the type.
  await page.getByRole("button", { name: /^Device type: choose another/ }).click();
  await page.getByRole("button", { name: /Stereo-X1/ }).click();
  await expect(page).toHaveURL(/scope=type/);
  await expect(
    page.getByRole("table", { name: "Units of the device type" }),
  ).toBeVisible();

  // Calibrate's seed row: a run whose seed does not fit is refused, said on
  // the row, and nothing is written (the bench runs are undistorted, this
  // project's camera model is not).
  await page.goto(`/#/calibrate?project=${enc(project)}`);
  const panel = page.getByRole("complementary", { name: "This run" });
  await expect(panel).toContainText("each camera starts from its own images");
  await panel.getByRole("combobox", { name: "Seed from a run of this project" }).click();
  await page.getByRole("option", { name: /run-002/ }).click();
  await expect(panel.getByText("The project cannot take this seed")).toBeVisible();
  await expect(panel).toContainText("each camera starts from its own images");
  const refused = await command(request, "open_project", { dir: project });
  expect((refused.body as { spec: { seed?: unknown } }).spec.seed).toBeUndefined();

  // Where the model fits, the row sets the seed, shows what it starts from,
  // clears it, and links to the device's Seed tab.
  const field = `${root}/stereo-sn0042-field`;
  const opened = (await command(request, "open_project", { dir: field }))
    .body as unknown as ProjectView;
  const undistorted = await command(request, "save_project", {
    dir: field,
    revision: opened.revision,
    spec: { ...opened.spec, config: { camera: { model: { distortion: "none" } } } },
  });
  expect(undistorted.status).toBe(200);
  await page.goto(`/#/calibrate?project=${enc(field)}`);
  const fieldPanel = page.getByRole("complementary", { name: "This run" });
  await fieldPanel
    .getByRole("combobox", { name: "Seed from a run of this project" })
    .click();
  await page.getByRole("option", { name: /run-001/ }).click();
  await expect(fieldPanel).toContainText("From a calibrated run");
  await expect(
    fieldPanel.getByRole("list", { name: "What the seed starts from" }),
  ).toContainText("cam0");
  await fieldPanel.getByRole("button", { name: "Clear" }).click();
  await expect(fieldPanel).toContainText("each camera starts from its own images");
  await fieldPanel
    .getByRole("link", { name: "Seed from the device or its type" })
    .click();
  await expect(page).toHaveURL(/#\/device\?.*tab=seed/);
  // The 422 is the refusal asked for above, not an error of the page.
  expect(errors.filter((e) => !e.includes("422 (Unprocessable"))).toEqual([]);

  const now = (await command(request, "open_project", { dir: field }))
    .body as unknown as ProjectView;
  const restored = await command(request, "save_project", {
    dir: field,
    revision: now.revision,
    spec: opened.spec,
  });
  expect(restored.status).toBe(200);
});
