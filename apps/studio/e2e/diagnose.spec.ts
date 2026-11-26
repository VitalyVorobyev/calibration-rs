import { expect, test } from "@playwright/test";
import { installTauriMock } from "./support/tauriMock";
import { ONE_PX_PNG_DATA_URL, PLANAR_EXPORT_FIXTURE } from "./support/fixtures";

/** Smoke test: Diagnose renders a fixture export loaded
 * through the mocked Tauri IPC layer. Exercises the full "Open
 * Export…" path end-to-end in a real browser: the dialog plugin's
 * `plugin:dialog|open`, `load_export` and `set_active_export` resolve
 * against the injected `window.__TAURI_INTERNALS__` shim rather than a
 * real backend, and the frame image (loaded through the asset protocol in
 * the app) resolves to a one-pixel PNG.
 */
test("Diagnose renders a fixture export loaded through mocked Tauri IPC", async ({
  page,
}) => {
  await installTauriMock(
    page,
    {
      "plugin:dialog|open": "/fake/export.json",
      load_export: { export: PLANAR_EXPORT_FIXTURE, export_dir: "/fake" },
      set_active_export: null,
    },
    ONE_PX_PNG_DATA_URL,
  );

  await page.goto("/");
  await page.getByRole("button", { name: "Open Export…" }).click();

  await expect(page.getByText("mean reproj: 0.220 px")).toBeVisible();
  await expect(page.getByRole("group", { name: "Frame viewer" })).toBeVisible();
  // The header badge switches from the generic subtitle to the loaded
  // export's classification (store/exportKind.ts).
  await expect(page.getByText("Planar intrinsics")).toBeVisible();
});
