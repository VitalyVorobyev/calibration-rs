/** Canonical calibration-export fixtures shared by Vitest component
 * tests and the Playwright e2e smoke specs.
 *
 * Extracted after the "planar" export fixture was hand-duplicated
 * across `e2e/support/fixtures.ts` and
 * `DiagnoseWorkspace/DiagnoseWorkspace.test.tsx` and had already
 * drifted (`projected_px: [10.2, 10.1]` vs `[10.1, 10.2]`). One
 * definition per concept: change the shape here and every consumer —
 * Vitest or Playwright — moves together instead of silently diverging.
 */
import type {
  ImageManifest,
  LaserFeatureResidual,
  TargetFeatureResidual,
} from "../types";

export const TARGET_MANIFEST: ImageManifest = {
  root: ".",
  frames: [{ pose: 0, camera: 0, path: "frame0.png" }],
};

export const LASER_MANIFEST: ImageManifest = {
  root: ".",
  frames: [
    { pose: 0, camera: 0, path: "frame0.png", kind: "target" },
    { pose: 0, camera: 0, path: "laser0.png", kind: "laser" },
  ],
};

export const TARGET_RESIDUAL: TargetFeatureResidual = {
  pose: 0,
  camera: 0,
  feature: 0,
  target_xyz_m: [0, 0, 0],
  observed_px: [10, 10],
  projected_px: [10.1, 10.2],
  error_px: 0.22,
};

export const LASER_RESIDUAL: LaserFeatureResidual = {
  pose: 0,
  camera: 0,
  feature: 0,
  observed_px: [5, 5],
  residual_m: 0.0002,
};

/** Minimal `PlanarIntrinsicsExport`-shaped payload — enough for
 * `detectExportKind` to classify it `"planar_intrinsics"` and for
 * `DiagnoseWorkspace` / the e2e smoke spec to have one frame and one
 * residual to draw. */
export const PLANAR_EXPORT_FIXTURE = {
  params: { camera: { sensor: { type: "identity" } } },
  per_feature_residuals: { target: [TARGET_RESIDUAL] },
  image_manifest: TARGET_MANIFEST,
  mean_reproj_error: 0.22,
};

/** A 1x1 transparent PNG data URL — a real browser (unlike jsdom) fully
 * decodes it, so `<img>`'s `load` event fires and `FrameCanvas` gets a
 * real `naturalWidth`/`naturalHeight` to draw against. */
export const ONE_PX_PNG_DATA_URL =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";
