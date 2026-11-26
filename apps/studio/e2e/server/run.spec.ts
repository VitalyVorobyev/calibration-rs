import { expect, test } from "@playwright/test";
import { command, libraryRoot, openLoggedIn, trackErrors } from "./support";

/** A calibration through the real server: the printed address opens the
 * project, the Run workspace starts the job, its progress arrives over the
 * event stream (SSE) and its end is shown. The fixture project has two
 * views, one short of what a rig calibration needs, so the run detects all
 * four images and then fails, as the library says: the job, its progress
 * and its typed outcome are what is under test. */

test("runs the project's calibration as a job with progress to its end", async ({
  page,
  request,
}) => {
  const errors = trackErrors(page);
  const root = await libraryRoot(request);
  const project = `${root}/rig-demo`;
  await openLoggedIn(page, `&project=${encodeURIComponent(project)}`);
  await expect(page.getByRole("tree", { name: "Saved runs" })).toBeVisible();

  await page.getByRole("link", { name: "Run calibration" }).click();
  const run = page.getByRole("button", { name: "Run", exact: true });
  await expect(run).toBeEnabled();
  await run.click();

  await expect(page.getByText(/Run failed \(session_set_input\)/)).toBeVisible({
    timeout: 60_000,
  });
  await expect(page.getByText(/insufficient data: need 3, got 2/)).toBeVisible();

  // The job's record on the server: its last progress was the detection of
  // all four images.
  const { body } = await command(request, "list_jobs", {});
  const jobs = body as unknown as {
    command: string;
    state: { kind: string };
    last_event?: { report?: { kind: string; done: number; total: number } };
  }[];
  const job = jobs.find((j) => j.command === "run_calibration");
  expect(job?.state.kind).toBe("done");
  expect(job?.last_event?.report).toMatchObject({ kind: "detection", done: 4, total: 4 });
  expect(errors).toEqual([]);
});
