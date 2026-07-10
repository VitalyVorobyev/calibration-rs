import { expect, test } from "@playwright/test";
import { installTauriMock } from "./support/tauriMock";
import { ONE_PX_PNG_DATA_URL, PLANAR_EXPORT_FIXTURE } from "./support/fixtures";

/** (c) of B-QUAL4-SMOKE: Diagnose renders a fixture export loaded
 * through the mocked Tauri IPC layer. Exercises the full "Open
 * Export…" path end-to-end in a real browser: the dialog plugin's
 * `plugin:dialog|open`, `load_export`, `set_active_export`, and
 * (once DiagnoseWorkspace mounts a frame) `load_image` all resolve
 * against the injected `window.__TAURI_INTERNALS__` shim rather than
 * a real backend.
 */
test("Diagnose renders a fixture export loaded through mocked Tauri IPC", async ({
  page,
}) => {
  await installTauriMock(page, {
    "plugin:dialog|open": "/fake/export.json",
    load_export: { export: PLANAR_EXPORT_FIXTURE, export_dir: "/fake" },
    set_active_export: null,
    load_image: ONE_PX_PNG_DATA_URL,
  });

  await page.goto("/");
  await page.getByRole("button", { name: "Open Export…" }).click();

  await expect(page.getByText("mean reproj: 0.220 px")).toBeVisible();
  await expect(page.locator("canvas")).toBeVisible();
  // The header badge switches from the generic subtitle to the loaded
  // export's classification (store/exportKind.ts).
  await expect(page.getByText("Planar intrinsics")).toBeVisible();
});
