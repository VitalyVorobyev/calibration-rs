import { expect, test } from "@playwright/test";
import { command, libraryRoot, openLoggedIn, trackErrors } from "./support";

/** A calibration through the real server: the path picker opens the
 * fixture project `planar-demo` (six rendered chessboard views), the Run
 * workspace runs it as the `calibrate` job with its progress over the
 * event stream (SSE), the run it writes is listed and shown, and the run's
 * label is set and the run deleted from the project navigator. */

test("runs a project as a job, lists the run, notes it and deletes it", async ({
  page,
  request,
}) => {
  const errors = trackErrors(page);
  const root = await libraryRoot(request);
  await openLoggedIn(page);

  await page.getByRole("button", { name: /Project/ }).click();
  await page.getByRole("menuitem", { name: "Open project…" }).click();
  const picker = page.getByRole("dialog", { name: "Open a project" });
  await picker.getByRole("option", { name: root }).dblclick();
  await picker.getByRole("option", { name: /planar-demo/ }).click();
  await picker.getByRole("button", { name: "Choose", exact: true }).click();
  await expect(picker).toHaveCount(0);

  await page.getByRole("link", { name: "Calibrate" }).click();
  await expect(page.getByText("No runs yet.")).toBeVisible();
  const run = page.getByRole("button", { name: "Run", exact: true });
  await expect(run).toBeEnabled();
  await run.click();
  await expect(page.getByText("Solve completed")).toBeVisible({ timeout: 60_000 });
  await expect(
    page.getByText(/6\/6 usable views · saved as runs\/run-001/),
  ).toBeVisible();

  // The run is listed, and shown in Inspect (the hand-off), with its pills
  // in the context bar and its job in the tray's indicator.
  const runs = page.getByRole("tree", { name: "Saved runs" });
  await expect(runs.getByText("run-001")).toBeVisible();
  await expect(page).toHaveURL(/#\/inspect\/frames\?project=.*&run=run-001$/);
  await expect(page.getByRole("combobox", { name: "Run" })).toContainText("run-001");
  await expect(page.getByText("Recorded", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: /^Jobs/ })).toBeVisible();
  await expect(page.getByText("Planar intrinsics")).toBeVisible();

  // The job's record on the server: its last progress came from the
  // library, and it ended with the run.
  const { body } = await command(request, "list_jobs", {});
  const jobs = body as unknown as {
    command: string;
    state: { kind: string; output?: { id: string } };
  }[];
  const job = jobs.find((j) => j.command === "calibrate");
  expect(job?.state.kind).toBe("done");
  expect(job?.state.output?.id).toBe("run-001");

  // A label, saved to the run's notes.
  const label = page.getByLabel("Run label");
  await label.fill("first light");
  await label.press("Enter");
  await expect(runs.getByText("run-001 · first light")).toBeVisible();
  const listed = await command(request, "list_runs", { dir: `${root}/planar-demo` });
  const listedRuns = listed.body as unknown as {
    id: string;
    notes: { label?: string };
  }[];
  expect(listedRuns[0]?.notes.label).toBe("first light");

  // Deleted, after the confirmation.
  await page.getByRole("button", { name: "Delete run" }).click();
  await page.getByRole("dialog").getByRole("button", { name: "Delete" }).click();
  await expect(page.getByText("No runs yet.")).toBeVisible();
  const after = await command(request, "list_runs", { dir: `${root}/planar-demo` });
  expect(after.body).toEqual([]);
  expect(errors).toEqual([]);
});

test("a run of the earlier project format fails as the library says", async ({
  request,
}) => {
  // The version 1 fixture has two views, one short of a rig calibration:
  // the job ends `failed` with the library's typed error.
  const root = await libraryRoot(request);
  const started = await command(request, "calibrate", { dir: `${root}/rig-demo` });
  const id = started.body.id as string;
  let state: { kind: string; error?: { kind: string; message: string } } = {
    kind: "queued",
  };
  for (let i = 0; i < 600 && ["queued", "running"].includes(state.kind); i++) {
    await new Promise((r) => setTimeout(r, 100));
    state = (await command(request, "job_status", { id })).body.state as typeof state;
  }
  expect(state.kind).toBe("failed");
  expect(state.error?.kind).toBe("calibration");
  expect(state.error?.message).toMatch(/need 3, got 2/);
});
