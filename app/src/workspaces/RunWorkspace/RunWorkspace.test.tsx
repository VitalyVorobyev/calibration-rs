// @vitest-environment jsdom
/** Happy-path component test for RunWorkspace (B-QUAL3-3): pick a
 * built-in preset, trigger Run, and assert the success banner renders
 * from a mocked `run_calibration_cmd` response.
 *
 * The Tauri IPC layer is mocked once, at the `invoke` seam, via
 * `@tauri-apps/api/mocks` (`mockIPC`) — no real backend, no disk I/O:
 * `load_text_file` returns a canned TOML manifest in place of the
 * preset's on-disk file, `default_config_cmd` returns a minimal
 * `PlanarIntrinsicsConfig`, and `run_calibration_cmd` returns a
 * synthetic success payload.
 */
import { clearMocks, mockIPC } from "@tauri-apps/api/mocks";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { RunWorkspace } from "./index";
import { useStore } from "../../store";

const MANIFEST_TOML = `
version = 1
topology = "planar_intrinsics"

[target]
kind = "chessboard"
rows = 9
cols = 6
square_size_m = 0.025

[[cameras]]
id = "cam0"

[cameras.images]
kind = "glob"
pattern = "*.png"
`;

const DEFAULT_CONFIG = {
  init: { init_iterations: 2, fix_k3: true, fix_tangential: false, zero_skew: true },
  solver: { max_iters: 50, verbosity: 0, robust_loss: "None" },
  distortion_model: "brown_conrady5",
  fix_camera: {
    intrinsics: { fx: false, fy: false, cx: false, cy: false },
    distortion: { k1: false, k2: false, k3: true, p1: false, p2: false },
  },
  fix_poses: [],
};

const RUN_SUCCESS_RESPONSE = {
  kind: "ok",
  export: {
    kind: "planar_intrinsics",
    per_feature_residuals: { target: [] },
    image_manifest: { root: ".", frames: [{ pose: 0, camera: 0, path: "frame0.png" }] },
    mean_reproj_error: 0.31,
  },
  duration_ms: 42,
  usable_views: 20,
  total_views: 20,
  cache_used: false,
};

beforeEach(() => {
  mockIPC((cmd) => {
    switch (cmd) {
      case "repo_root_cmd":
        return "/fake/repo/root";
      case "load_text_file":
        return MANIFEST_TOML;
      case "default_config_cmd":
        return DEFAULT_CONFIG;
      case "run_calibration_cmd":
        return RUN_SUCCESS_RESPONSE;
      default:
        return null;
    }
  });
});

afterEach(() => {
  clearMocks();
  cleanup();
  useStore.getState().resetExport();
});

/** Finds the "stereo-left" preset card and returns its "Use preset" button. */
function stereoLeftUseButton(): HTMLElement {
  const name = screen.getByText("Stereo · cam-left");
  const card = name.closest(".rounded-lg");
  if (!card) throw new Error("preset card container not found");
  return within(card as HTMLElement).getByRole("button", { name: "Use preset" });
}

describe("RunWorkspace happy path", () => {
  it("selects a preset, runs, and renders the mocked success response", async () => {
    render(
      <MemoryRouter>
        <RunWorkspace />
      </MemoryRouter>,
    );

    fireEvent.click(stereoLeftUseButton());

    // Preset load resolves asynchronously (load_text_file + TOML parse +
    // default_config_cmd); the quick-start grid collapses to the active-
    // preset bar once state settles.
    await screen.findByText("Stereo · cam-left", {}, { timeout: 2000 });
    const runButton = await screen.findByRole("button", { name: "Run" });
    expect(runButton.hasAttribute("disabled")).toBe(false);

    fireEvent.click(runButton);

    const banner = await screen.findByText("Solve completed", {}, { timeout: 2000 });
    expect(banner).toBeTruthy();
    expect(screen.getByText(/42 ms/)).toBeTruthy();
    expect(screen.getByText(/20\/20 usable views/)).toBeTruthy();

    // The success payload is threaded into the shared store, ready for
    // the /diagnose handoff.
    expect(useStore.getState().data).toEqual(RUN_SUCCESS_RESPONSE.export);
  });
});

describe("RunWorkspace progress + cancellation", () => {
  it("shows the stage checklist for a running solve and cancels it", async () => {
    // Override the default mock: `run_calibration_cmd` stays pending
    // until `cancel_run_cmd` resolves it with a cancelled response — so
    // the running state (and its Cancel button) is observable.
    let resolveRun!: (value: unknown) => void;
    mockIPC((cmd) => {
      switch (cmd) {
        case "repo_root_cmd":
          return "/fake/repo/root";
        case "load_text_file":
          return MANIFEST_TOML;
        case "default_config_cmd":
          return DEFAULT_CONFIG;
        case "run_calibration_cmd":
          return new Promise((res) => {
            resolveRun = res;
          });
        case "cancel_run_cmd":
          resolveRun({ kind: "cancelled" });
          return true;
        default:
          return null;
      }
    });

    render(
      <MemoryRouter>
        <RunWorkspace />
      </MemoryRouter>,
    );

    fireEvent.click(stereoLeftUseButton());
    const runButton = await screen.findByRole("button", { name: "Run" });
    fireEvent.click(runButton);

    // The progress checklist renders all three stages plus a Cancel
    // button while the solve is in flight.
    const cancelButton = await screen.findByRole("button", { name: "Cancel" });
    expect(screen.getByText("Detecting features")).toBeTruthy();
    expect(screen.getByText("Solving")).toBeTruthy();
    expect(screen.getByText("Exporting")).toBeTruthy();

    fireEvent.click(cancelButton);

    // Terminal cancelled state — a distinct banner, not an error.
    await screen.findByText("Run cancelled", {}, { timeout: 2000 });
    // No export was committed to the store.
    expect(useStore.getState().data).toBeNull();
  });
});
