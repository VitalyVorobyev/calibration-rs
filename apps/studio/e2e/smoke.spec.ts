import { expect, type Page, test } from "@playwright/test";

/** The bun-only smoke run: the app boots with no back end answering (a
 * plain `bun run dev` tab, its `/api` proxy pointing nowhere), says so, and
 * every screen's address mounts without throwing or logging a console error
 * of its own. The browser's own report of the failed `/api` requests is the
 * scenario, not an error of the app's. */

function trackConsoleErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error" && !msg.text().startsWith("Failed to load resource")) {
      errors.push(msg.text());
    }
  });
  page.on("pageerror", (err) => errors.push(err.message));
  return errors;
}

test("app boots, the shell renders and says there is no back end", async ({ page }) => {
  const errors = trackConsoleErrors(page);

  await page.goto("/");

  await expect(page.getByRole("navigation", { name: "Screens" })).toBeVisible();
  // The index route redirects to the Library (router.tsx).
  await expect(page).toHaveURL(/#\/library$/);
  await expect(page.getByText("No back end answers")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Library" })).toBeVisible();

  expect(errors).toEqual([]);
});

const ADDRESSES: { hash: string; marker: RegExp }[] = [
  { hash: "#/library", marker: /^Library$/ },
  // The Device screen reads the library, so it opens with no project.
  { hash: "#/device", marker: /^Device$/ },
  // Screens that work on a project say so while none is open.
  { hash: "#/dataset", marker: /Dataset works on a project/ },
  { hash: "#/detect", marker: /Detect works on a project/ },
  { hash: "#/calibrate", marker: /Calibrate works on a project/ },
  { hash: "#/inspect/frames", marker: /Inspect works on a project/ },
  // Screens that are not built say so instead of breaking.
  { hash: "#/compare", marker: /Coming in a later release/ },
  // The addresses of the screens before the redesign still open.
  { hash: "#/diagnose", marker: /Inspect works on a project/ },
  { hash: "#/run", marker: /Calibrate works on a project/ },
];

for (const address of ADDRESSES) {
  test(`mounts ${address.hash} without console errors`, async ({ page }) => {
    const errors = trackConsoleErrors(page);

    await page.goto(`/${address.hash}`);

    await expect(page.getByText(address.marker).first()).toBeVisible();
    await expect(page.getByText("No back end answers")).toBeVisible();

    expect(errors).toEqual([]);
  });
}
