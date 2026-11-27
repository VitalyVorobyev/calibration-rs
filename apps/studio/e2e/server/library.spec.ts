import { expect, test } from "@playwright/test";
import { command, imageWidth, libraryRoot, openLoggedIn, trackErrors } from "./support";

/** The fixture library through the in-app path picker: open the project,
 * open its run, see its frames (images through the `image` resource); and
 * nothing outside the root. */

test("opens a project through the path picker and shows its run", async ({
  page,
  request,
}) => {
  const errors = trackErrors(page);
  await openLoggedIn(page);

  await page.getByRole("button", { name: /Project/ }).click();
  await page.getByRole("menuitem", { name: "Open project…" }).click();
  const picker = page.getByRole("dialog", { name: "Open a project" });
  await expect(picker).toBeVisible();
  // The roots first (sorted by path, so the library's place among them
  // depends on the machine); into the library; the project is marked and chosen.
  await picker.getByRole("option", { name: await libraryRoot(request) }).dblclick();
  await picker.getByRole("option", { name: /rig-demo/ }).click();
  await picker.getByRole("button", { name: "Choose", exact: true }).click();
  await expect(picker).toHaveCount(0);

  // The project's one run is the current one: Inspect shows it.
  await page.getByRole("link", { name: "Inspect" }).click();
  const runs = page.getByRole("tree", { name: "Saved runs" });
  await expect(runs).toBeVisible();
  await expect(page.getByRole("combobox", { name: "Run" })).toContainText("run-001");
  await runs.getByText("run-001").click();
  await expect(page.getByText("Rig extrinsics")).toBeVisible();
  await expect(page.getByText("features 20")).toBeVisible();
  expect(await imageWidth(page, '[aria-label="Frame viewer"]')).toBe(160);

  // The epipolar pair: undistorted images from the server.
  await page.getByRole("tab", { name: "Epipolar" }).click();
  expect(await imageWidth(page, "section", 0)).toBe(160);
  expect(errors).toEqual([]);
});

test("the project is a recent one afterwards, on the server", async ({ request }) => {
  const root = await libraryRoot(request);
  await command(request, "open_project", { dir: `${root}/rig-demo` });
  const { body } = await command(request, "recent_projects", {});
  expect((body as unknown as string[])[0]).toBe(`${root}/rig-demo`);
});

test("paths outside the root are refused", async ({ request }) => {
  const root = await libraryRoot(request);
  for (const [name, args] of [
    ["list_dir", { path: `${root}/..` }],
    ["list_dir", { path: "/" }],
    ["read_manifest", { path: "/etc/hosts" }],
    ["open_project", { dir: `${root}/rig-demo/../..` }],
  ] as const) {
    const { status, body } = await command(request, name, args);
    expect(status, `${name} ${JSON.stringify(args)}`).toBe(403);
    expect(body.kind).toBe("forbidden_path");
  }
  const image = await request.get(
    `/api/r/image?q=${encodeURIComponent(JSON.stringify({ path: `${root}/../token` }))}`,
    { headers: { Authorization: "Bearer e2e-token-0123456789abcdefghij" } },
  );
  expect(image.status()).toBe(403);
});
