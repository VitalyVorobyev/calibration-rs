// @vitest-environment jsdom
/** Component tests for DiagnoseWorkspace (B-QUAL3-2): render the
 * workspace once per calibration export kind (the eight `ExportKind`
 * discriminants from `store/exportKind.ts`) and assert it mounts
 * without throwing.
 *
 * Fixtures are minimal objects satisfying `detectExportKind`'s
 * discriminating fields (mirroring `exportKind.test.ts`), each carrying
 * just enough of `AnyExport` — a one-frame `image_manifest` and a
 * `per_feature_residuals` block — for `DiagnoseWorkspace` to have
 * something to draw. The Tauri IPC layer is mocked at the `invoke` seam
 * via `@tauri-apps/api/mocks` (`mockIPC`) so `useImageData`'s
 * `load_image` call resolves instead of throwing outside a real Tauri
 * runtime.
 */
import { clearMocks, mockIPC } from "@tauri-apps/api/mocks";
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { DiagnoseWorkspace } from "./index";
import { useStore } from "../../store";
import { detectExportKind, type ExportKind } from "../../store/exportKind";
import type {
  ImageManifest,
  LaserFeatureResidual,
  TargetFeatureResidual,
} from "../../types";

const TARGET_MANIFEST: ImageManifest = {
  root: ".",
  frames: [{ pose: 0, camera: 0, path: "frame0.png" }],
};

const LASER_MANIFEST: ImageManifest = {
  root: ".",
  frames: [
    { pose: 0, camera: 0, path: "frame0.png", kind: "target" },
    { pose: 0, camera: 0, path: "laser0.png", kind: "laser" },
  ],
};

const TARGET_RESIDUAL: TargetFeatureResidual = {
  pose: 0,
  camera: 0,
  feature: 0,
  target_xyz_m: [0, 0, 0],
  observed_px: [10, 10],
  projected_px: [10.1, 10.2],
  error_px: 0.22,
};

const LASER_RESIDUAL: LaserFeatureResidual = {
  pose: 0,
  camera: 0,
  feature: 0,
  observed_px: [5, 5],
  residual_m: 0.0002,
};

// One minimal fixture per `ExportKind`, grounded in the discriminating
// fields `detectExportKind` actually inspects (see exportKind.ts).
const KIND_FIXTURES: Record<Exclude<ExportKind, "unknown">, unknown> = {
  planar_intrinsics: {
    params: { camera: { sensor: { type: "identity" } } },
    per_feature_residuals: { target: [TARGET_RESIDUAL] },
    image_manifest: TARGET_MANIFEST,
    mean_reproj_error: 0.22,
  },
  scheimpflug_intrinsics: {
    params: { camera: { sensor: { type: "scheimpflug" } } },
    per_feature_residuals: { target: [TARGET_RESIDUAL] },
    image_manifest: TARGET_MANIFEST,
    mean_reproj_error: 0.22,
  },
  single_cam_handeye: {
    camera: {},
    handeye_mode: "EyeInHand",
    per_feature_residuals: { target: [TARGET_RESIDUAL] },
    image_manifest: TARGET_MANIFEST,
    mean_reproj_error: 0.22,
  },
  laserline_device: {
    estimate: {},
    stats: {},
    per_feature_residuals: { target: [], laser: [LASER_RESIDUAL] },
    image_manifest: LASER_MANIFEST,
  },
  rig_extrinsics: {
    cameras: [{}],
    cam_se3_rig: [],
    rig_se3_target: [],
    per_feature_residuals: { target: [TARGET_RESIDUAL] },
    image_manifest: TARGET_MANIFEST,
    mean_reproj_error: 0.22,
  },
  rig_handeye: {
    cameras: [{}],
    handeye_mode: "EyeToHand",
    per_feature_residuals: { target: [TARGET_RESIDUAL] },
    image_manifest: TARGET_MANIFEST,
    mean_reproj_error: 0.22,
  },
  rig_handeye_laserline: {
    laser_planes_rig: [],
    handeye_mode: "EyeInHand",
    cameras: [{}],
    per_feature_residuals: { target: [TARGET_RESIDUAL], laser: [LASER_RESIDUAL] },
    image_manifest: LASER_MANIFEST,
    mean_reproj_error: 0.22,
  },
  rig_laserline_device: {
    laser_planes_rig: [],
    laser_planes_cam: [],
    per_feature_residuals: { target: [], laser: [LASER_RESIDUAL] },
    image_manifest: LASER_MANIFEST,
  },
};

beforeEach(() => {
  // The only IPC call DiagnoseWorkspace's subtree makes on mount is
  // `useImageData`'s `load_image`; every fixture's manifest points at a
  // fake path that no real Tauri backend is behind. jsdom never
  // actually decodes the data URL (no <img> network stack), so the
  // exact payload doesn't matter — it just needs to resolve rather
  // than reject (see `isTauriContext`'s guard note in lib/tauri.ts).
  mockIPC(() => "data:image/png;base64,AAAA");
});

afterEach(() => {
  clearMocks();
  cleanup();
  useStore.getState().resetExport();
});

describe("DiagnoseWorkspace", () => {
  it("renders the empty state when no export is loaded", () => {
    render(<DiagnoseWorkspace />);
    expect(screen.getByText(/Open an/)).toBeTruthy();
  });

  for (const [kind, fixture] of Object.entries(KIND_FIXTURES) as [
    Exclude<ExportKind, "unknown">,
    unknown,
  ][]) {
    it(`renders without crashing for export kind "${kind}"`, () => {
      // Sanity: the fixture actually round-trips through the same
      // classifier the store uses, so a fixture/kind mismatch fails
      // loudly here rather than silently under-testing a branch.
      expect(detectExportKind(fixture)).toBe(kind);

      useStore.getState().acceptLiveRunExport(fixture, "/fake/export/dir");
      expect(useStore.getState().loadError).toBeNull();

      const { container } = render(<DiagnoseWorkspace />);
      expect(container.querySelector("canvas")).toBeTruthy();
    });
  }
});
