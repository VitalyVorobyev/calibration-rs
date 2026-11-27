// @vitest-environment jsdom
/** Component tests for DiagnoseWorkspace (trimmed by a later
 * review pass): render the workspace against the two shapes that
 * actually change its render branches, plus the empty state.
 *
 * `DiagnoseWorkspace` never inspects `ExportKind` itself — only
 * `frames` / `laserFrames` / `per_feature_residuals` (see `index.tsx`) —
 * so mounting it once per one of the eight `ExportKind` discriminants
 * exercised the same two branches eight times over. The eight-way
 * classification itself stays covered by `exportKind.test.ts`; here we
 * only need one target-manifest export (no laser view available) and
 * one laser-manifest export (laser view + laser residuals).
 *
 * Fixtures are shared with the Playwright e2e specs via
 * `src/test/exportFixtures.ts` (see that module's header). The back end
 * is a mock transport (`api/transport/mock.ts`), the desktop app's kind, so
 * `useImageData`'s image URL resolves without a Tauri runtime.
 */
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { setTransport } from "../../api";
import { createMockTransport } from "../../api/transport/mock";
import { DiagnoseWorkspace } from "./index";
import { useStore } from "../../store";
import { detectExportKind } from "../../store/exportKind";
import {
  LASER_MANIFEST,
  LASER_RESIDUAL,
  PLANAR_EXPORT_FIXTURE,
} from "../../test/exportFixtures";

// Target-manifest export: classifies as planar_intrinsics, one target
// frame, no laser data — exercises DiagnoseWorkspace's default branch.
const TARGET_MANIFEST_EXPORT = PLANAR_EXPORT_FIXTURE;

// Laser-manifest export: classifies as laserline_device, carries a
// laser-kind frame + laser residuals — exercises the laser-view branch
// (`hasLaser` in index.tsx).
const LASER_MANIFEST_EXPORT = {
  kind: "laserline_device",
  stats: {},
  diagnostics: {
    per_feature_residuals: { target: [], laser: [LASER_RESIDUAL] },
    image_manifest: LASER_MANIFEST,
  },
};

beforeEach(() => {
  // `useImageData` loads each frame by its URL; jsdom never fetches it,
  // so the mock's stand-in will do. No command is expected on mount, but
  // the mock makes the back end the desktop app's, as in the app.
  setTransport(createMockTransport());
});

afterEach(() => {
  setTransport(null);
  cleanup();
  useStore.getState().resetExport();
});

/** The run the fixtures stand for. */
const RUN = { dir: "/fake", id: "run-001" };

describe("DiagnoseWorkspace", () => {
  it("renders the empty state when no export is loaded", () => {
    render(<DiagnoseWorkspace />);
    expect(screen.getByText(/Open a project/)).toBeTruthy();
  });

  it("renders a target-manifest export (planar_intrinsics)", () => {
    // Sanity: the fixture actually round-trips through the same
    // classifier the store uses.
    expect(detectExportKind(TARGET_MANIFEST_EXPORT)).toBe("planar_intrinsics");

    useStore
      .getState()
      .showRun(RUN, { export: TARGET_MANIFEST_EXPORT, image_root: "/fake/export/dir" });
    expect(useStore.getState().loadError).toBeNull();

    const { container } = render(<DiagnoseWorkspace />);
    expect(
      container.querySelector('[role="group"][aria-label="Frame viewer"]'),
    ).toBeTruthy();
  });

  it("navigates an export without an image list by its residuals", () => {
    const diagnostics = {
      per_feature_residuals: PLANAR_EXPORT_FIXTURE.diagnostics.per_feature_residuals,
    };
    useStore.getState().showRun(RUN, {
      export: { ...PLANAR_EXPORT_FIXTURE, diagnostics },
      image_root: null,
    });
    const state = useStore.getState();
    expect(state.loadError).toBeNull();
    expect(state.frames).toEqual([
      { pose: 0, camera: 0, label: "pose 0 · cam 0", abs_path: null },
    ]);

    const { container } = render(<DiagnoseWorkspace />);
    expect(screen.getByText(/This export has no image list/)).toBeTruthy();
    expect(container.querySelector('[aria-label="Frame viewer"]')).toBeNull();
    // The frame's residual legend is still there; the image histogram is not.
    expect(screen.getByText("mean 0.220 px")).toBeTruthy();
    expect(screen.queryByText("histogram")).toBeNull();
  });

  it("renders a laser-manifest export (laserline_device)", () => {
    expect(detectExportKind(LASER_MANIFEST_EXPORT)).toBe("laserline_device");

    useStore
      .getState()
      .showRun(RUN, { export: LASER_MANIFEST_EXPORT, image_root: "/fake/export/dir" });
    expect(useStore.getState().loadError).toBeNull();

    const { container } = render(<DiagnoseWorkspace />);
    expect(
      container.querySelector('[role="group"][aria-label="Frame viewer"]'),
    ).toBeTruthy();
  });
});
