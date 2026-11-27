import { expect, test, type Page } from "@playwright/test";
import { command, libraryRoot, openLoggedIn, trackErrors } from "./support";

/** The Library screen and the New project and Import dialogs on the real
 * server and the fixture library: the projects by device type and device,
 * a project made of a folder of images (the library infers its layout), a
 * project opened from the Library, and an export imported as a run. */

/** Choose `folder` (a name under the library root) in the open path picker. */
async function pick(page: Page, title: string, root: string, folder: string) {
  const picker = page.getByRole("dialog", { name: title });
  await picker.getByRole("option", { name: root }).dblclick();
  await picker.getByRole("option", { name: new RegExp(`^${folder}`) }).click();
  await picker.getByRole("button", { name: "Choose", exact: true }).click();
  await expect(picker).toHaveCount(0);
}

test("lists the projects by device type and device, and searches them", async ({
  page,
}) => {
  const errors = trackErrors(page);
  await openLoggedIn(page);
  await page.goto("/#/library");
  const tree = page.getByRole("tree", { name: "Projects by device" });
  await expect(tree).toBeVisible();
  await expect(tree.getByRole("treeitem", { name: /Stereo-X1/ })).toBeVisible();
  await expect(
    tree.getByRole("treeitem", { name: /Bench cell 3 · SN-0042/ }),
  ).toBeVisible();
  await expect(tree.getByRole("treeitem", { name: /\(no device\)/ })).toBeVisible();

  // The latest run of each: its verdict in words and its RMS.
  const table = page.getByRole("table", { name: "Projects" });
  const second = table.getByRole("row", { name: /Second unit/ });
  await expect(second).toContainText("Fail");
  await expect(second).toContainText("1.958");

  await page.getByRole("searchbox", { name: "Search projects" }).fill("sn-0057");
  await expect(table.getByRole("row", { name: /Second unit/ })).toBeVisible();
  await expect(table.getByRole("row", { name: /Bench calibration/ })).toHaveCount(0);
  // The search is in the address, so a link reopens it.
  await expect(page).toHaveURL(/q=sn-0057/);
  expect(errors).toEqual([]);
});

test("opens a project from the Library on Inspect, and one with no run on Calibrate", async ({
  page,
  request,
}) => {
  await openLoggedIn(page);
  await page.goto("/#/library");
  const table = page.getByRole("table", { name: "Projects" });
  await table
    .getByRole("row", { name: /Bench calibration/ })
    .getByRole("button", { name: "Open" })
    .click();
  await expect(page).toHaveURL(/#\/inspect\/frames\?project=.*stereo-sn0042/);
  await expect(page.getByRole("button", { name: "Project", exact: true })).toContainText(
    "Bench calibration",
  );
  // The device is in the breadcrumb, from the project file.
  await expect(page.getByRole("navigation", { name: "Project" })).toContainText(
    "Stereo-X1",
  );

  await page.goto("/#/library");
  await table
    .getByRole("row", { name: /Four-camera cell/ })
    .getByRole("button", { name: "Open" })
    .click();
  await expect(page).toHaveURL(/#\/calibrate\?project=.*rig-four-cam/);
  const root = await libraryRoot(request);
  const { body } = await command(request, "recent_projects", {});
  expect((body as unknown as string[])[0]).toBe(`${root}/rig-four-cam`);
});

test("creates a project from a folder of images through the five steps", async ({
  page,
  request,
}) => {
  const errors = trackErrors(page);
  const root = await libraryRoot(request);
  await openLoggedIn(page);
  await page.goto("/#/library");
  await page.getByRole("button", { name: "New project" }).click();
  const dialog = page.getByRole("dialog", { name: "New project" });

  // 1. The folder: only read, its layout inferred.
  await dialog.getByRole("button", { name: /A folder of images/ }).click();
  await pick(page, "Pick the folder of images", root, "captures-pair");
  await expect(dialog.getByText("2 cameras, 2 to decide in the next step")).toBeVisible();
  await dialog.getByRole("button", { name: "Next" }).click();

  // 2. The layout: two cameras of four views as a matrix of thumbnails, and
  // the one thing the sniffer left open, asked in words.
  await expect(dialog.getByText("1 thing the layout does not say")).toBeVisible();
  await expect(
    dialog.getByRole("grid", { name: "First views of the new project" }),
  ).toBeVisible();
  await expect(dialog.getByRole("button", { name: "Next" })).toBeDisabled();
  await dialog.getByRole("button", { name: /Confirm: What are you calibrating/ }).click();
  await expect(dialog.getByRole("button", { name: "Next" })).toBeEnabled();
  await dialog.getByRole("button", { name: "Next" }).click();

  // 3. The target.
  await dialog.getByRole("spinbutton", { name: /rows/i }).first().fill("4");
  await dialog.getByRole("spinbutton", { name: /cols/i }).first().fill("5");
  await dialog
    .getByRole("spinbutton", { name: /square size/i })
    .first()
    .fill("0.02");
  await dialog.getByRole("button", { name: "Next" }).click();

  // 4. The device (optional).
  await dialog.getByLabel("Device type").fill("Stereo-X1");
  await dialog.getByLabel("Device id").fill("SN-0777");
  await dialog.getByRole("button", { name: "Next" }).click();

  // 5. Name and place; nothing was written until now.
  await expect(dialog.getByLabel("Project name")).toHaveValue("captures-pair");
  const before = await command(request, "open_project", {
    dir: `${root}/Stereo-X1/captures-pair`,
  });
  expect(before.status).not.toBe(200);
  await dialog.getByRole("button", { name: "Create project" }).click();
  await expect(dialog).toHaveCount(0);

  const made = await command(request, "open_project", {
    dir: `${root}/Stereo-X1/captures-pair`,
  });
  expect(made.status).toBe(200);
  const spec = made.body.spec as { device: { id: string }; dataset: { root: string } };
  expect(spec.device.id).toBe("SN-0777");
  expect(spec.dataset.root).toBe("../../captures-pair");
  // It is in the Library, under its device.
  await page.goto("/#/library");
  const tree = page.getByRole("tree", { name: "Projects by device" });
  await expect(tree.getByRole("treeitem", { name: "SN-0777" })).toBeVisible();
  expect(errors).toEqual([]);
});

test("refuses a project folder that already holds a project, keeping what was entered", async ({
  page,
  request,
}) => {
  const root = await libraryRoot(request);
  await openLoggedIn(page);
  await page.goto("/#/library");
  await page.getByRole("button", { name: "New project" }).click();
  const dialog = page.getByRole("dialog", { name: "New project" });
  await dialog.getByRole("button", { name: /An existing dataset manifest/ }).click();
  // The manifest of the old format's project: a bare manifest file.
  const picker = page.getByRole("dialog", { name: /Pick a dataset/ });
  await picker.getByRole("option", { name: root }).dblclick();
  await picker.getByRole("option", { name: /^rig-demo/ }).dblclick();
  await picker.getByRole("option", { name: /^dataset.json/ }).click();
  await picker.getByRole("button", { name: "Choose", exact: true }).click();
  await expect(dialog.getByText(/2 cameras/)).toBeVisible();
  for (let i = 0; i < 4; i++) {
    await dialog.getByRole("button", { name: /^(Next|Set later)$/ }).click();
  }
  // Into a folder that is a project already.
  await dialog
    .getByRole("textbox", { name: "Project folder" })
    .fill(`${root}/rig-four-cam`);
  await dialog.getByRole("button", { name: "Create project" }).click();
  await expect(dialog.getByText(/already holds a project/)).toBeVisible();
  await expect(dialog.getByLabel("Project name")).toHaveValue("rig-demo");
  await dialog.getByRole("button", { name: "Cancel" }).click();
  await page.getByRole("button", { name: "Discard" }).click();
});

test("imports an export as a run and opens it in Inspect", async ({ page, request }) => {
  const root = await libraryRoot(request);
  await openLoggedIn(page);
  await page.goto("/#/library");
  await page.getByRole("button", { name: "Import export" }).click();
  const dialog = page.getByRole("dialog", { name: "Import an export" });
  await dialog.getByLabel("Export file or run folder").fill(`${root}/vcal-run`);
  await dialog.getByRole("combobox", { name: "Project" }).click();
  await page.getByRole("option", { name: /Second unit/ }).click();
  await dialog.getByLabel("Run label").fill("vendor tool");
  await dialog.getByRole("button", { name: "Import", exact: true }).click();
  await expect(dialog.getByText(/Imported as run-002/)).toBeVisible();
  await dialog.getByRole("button", { name: "Open in Inspect" }).click();
  await expect(page).toHaveURL(/#\/inspect\/frames\?project=.*stereo-sn0057&run=run-002/);
  const runs = await command(request, "list_runs", { dir: `${root}/stereo-sn0057` });
  const list = runs.body as unknown as {
    id: string;
    notes: { label?: string };
    state: { kind: string; source: { kind: string } };
  }[];
  expect(list.map((r) => r.id)).toEqual(["run-001", "run-002"]);
  expect(list[1].notes.label).toBe("vendor tool");
  expect(list[1].state.source.kind).toBe("imported");
});

test("an export of another workflow is refused in the dialog, with the project still chosen", async ({
  page,
  request,
}) => {
  const root = await libraryRoot(request);
  await openLoggedIn(page);
  await page.goto("/#/library");
  await page.getByRole("button", { name: "Import export" }).click();
  const dialog = page.getByRole("dialog", { name: "Import an export" });
  await dialog.getByLabel("Export file or run folder").fill(`${root}/vcal-run`);
  await dialog.getByRole("combobox", { name: "Project" }).click();
  await page.getByRole("option", { name: /planar-demo/ }).click();
  await dialog.getByRole("button", { name: "Import", exact: true }).click();
  await expect(dialog.getByText(/does not belong to that project/)).toBeVisible();
  await expect(dialog.getByRole("combobox", { name: "Project" })).toContainText(
    "planar-demo",
  );
});
