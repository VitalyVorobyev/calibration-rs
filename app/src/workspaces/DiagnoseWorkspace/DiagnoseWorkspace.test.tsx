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
 * `src/test/exportFixtures.ts` (see that module's header). The Tauri
 * IPC layer is mocked at the `invoke` seam via `@tauri-apps/api/mocks`
 * (`mockIPC`, `mockConvertFileSrc`) so `useImageData`'s asset URL resolves instead of
 * throwing outside a real Tauri runtime.
 */
import { clearMocks, mockConvertFileSrc, mockIPC } from "@tauri-apps/api/mocks";
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
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
  params: {},
  stats: {},
  diagnostics: {
    per_feature_residuals: { target: [], laser: [LASER_RESIDUAL] },
    image_manifest: LASER_MANIFEST,
  },
};

beforeEach(() => {
  // `useImageData` loads each frame through the asset protocol
  // (`convertFileSrc`); jsdom never fetches the URL, so any mapping
  // will do. No IPC call is expected on mount, but the mock keeps
  // `isTauriContext` true as in the app.
  mockIPC(() => null);
  mockConvertFileSrc("macos");
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

  it("renders a target-manifest export (planar_intrinsics)", () => {
    // Sanity: the fixture actually round-trips through the same
    // classifier the store uses.
    expect(detectExportKind(TARGET_MANIFEST_EXPORT)).toBe("planar_intrinsics");

    useStore.getState().acceptLiveRunExport(TARGET_MANIFEST_EXPORT, "/fake/export/dir");
    expect(useStore.getState().loadError).toBeNull();

    const { container } = render(<DiagnoseWorkspace />);
    expect(
      container.querySelector('[role="group"][aria-label="Frame viewer"]'),
    ).toBeTruthy();
  });

  it("renders a laser-manifest export (laserline_device)", () => {
    expect(detectExportKind(LASER_MANIFEST_EXPORT)).toBe("laserline_device");

    useStore.getState().acceptLiveRunExport(LASER_MANIFEST_EXPORT, "/fake/export/dir");
    expect(useStore.getState().loadError).toBeNull();

    const { container } = render(<DiagnoseWorkspace />);
    expect(
      container.querySelector('[role="group"][aria-label="Frame viewer"]'),
    ).toBeTruthy();
  });
});
