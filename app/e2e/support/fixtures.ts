/** Minimal fixture export payloads for the Playwright smoke specs.
 * Mirrors the `AnyExport` shapes exercised by the Vitest component
 * tests (`src/workspaces/DiagnoseWorkspace/DiagnoseWorkspace.test.tsx`)
 * — a `planar_intrinsics` classification (see `detectExportKind`) is
 * enough to prove the mocked-IPC → store → Diagnose render path works
 * end-to-end in a real browser.
 */
export const PLANAR_EXPORT_FIXTURE = {
  params: { camera: { sensor: { type: "identity" } } },
  per_feature_residuals: {
    target: [
      {
        pose: 0,
        camera: 0,
        feature: 0,
        target_xyz_m: [0, 0, 0],
        observed_px: [10, 10],
        projected_px: [10.2, 10.1],
        error_px: 0.22,
      },
    ],
  },
  image_manifest: {
    root: ".",
    frames: [{ pose: 0, camera: 0, path: "frame0.png" }],
  },
  mean_reproj_error: 0.22,
};

/** A 1×1 transparent PNG data URL — a real browser (unlike jsdom) fully
 * decodes it, so `<img>`'s `load` event fires and `FrameCanvas` gets a
 * real `naturalWidth`/`naturalHeight` to draw against. */
export const ONE_PX_PNG_DATA_URL =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";
