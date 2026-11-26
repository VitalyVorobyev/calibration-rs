// @vitest-environment jsdom
/** Happy-path component test for RunWorkspace: pick a
 * built-in preset, trigger Run, and assert the success banner renders
 * from a mocked `run_calibration_cmd` response.
 *
 * The back end is a mock transport — no real back end, no disk I/O:
 * the `fetch_sample` job answers a fake manifest path, `read_manifest` a
 * canned manifest in place of the preset's on-disk file, `default_config`
 * a minimal `PlanarIntrinsicsConfig`, and the `run_calibration` job a
 * synthetic success payload.
 */
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { TooltipProvider } from "@vitavision/ui";
import { MemoryRouter } from "react-router";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { setTransport, StudioError } from "../../api";
import type { DatasetSpec, RunResponse } from "../../api/generated/types";
import {
  type CommandHandlers,
  createMockTransport,
  type JobHandlers,
  type MockTransport,
} from "../../api/transport/mock";
import { RunWorkspace } from "./index";
import { useStore } from "../../store";
import { resetRunSetup } from "../../store/runSetup";
import { useProject } from "../../project/projectStore";

// The manifest as the back end reads it from the preset's TOML.
const MANIFEST = {
  version: 1,
  topology: "planar_intrinsics",
  target: { kind: "chessboard", rows: 9, cols: 6, square_size_m: 0.025 },
  cameras: [{ id: "cam0", images: { kind: "glob", pattern: "*.png" } }],
} as unknown as DatasetSpec;

const DEFAULT_CONFIG = {
  intrinsics_init: { iterations: 2, fix_k3: true },
  solver: {
    max_iters: 50,
    verbosity: 0,
    backend: "tiny_solver",
  },
  camera: {
    model: { projection: "pinhole", distortion: "brown_conrady5", sensor: "identity" },
    fix: {},
  },
  optimize: { reprojection_loss: { kind: "none" }, fix_poses: [] },
};

const RUN_SUCCESS_RESPONSE: RunResponse = {
  kind: "ok",
  export: {
    kind: "planar_intrinsics",
    diagnostics: {
      per_feature_residuals: { target: [] },
      image_manifest: { root: ".", frames: [{ view: 0, camera: 0, path: "frame0.png" }] },
    },
    mean_reproj_error: 0.31,
  },
  duration_ms: 42,
  usable_views: 20,
  total_views: 20,
  cache_used: false,
};

/** The setup commands every test needs, plus `commands` and `jobs`. */
function backend(commands: CommandHandlers = {}, jobs: JobHandlers = {}): MockTransport {
  const mock = createMockTransport({
    commands: {
      read_manifest: () => MANIFEST,
      default_config: () => DEFAULT_CONFIG,
      ...commands,
    },
    jobs: {
      fetch_sample: () => "/fake/cache/kuka_1/dataset.toml",
      run_calibration: () => RUN_SUCCESS_RESPONSE,
      ...jobs,
    },
  });
  setTransport(mock);
  return mock;
}

beforeEach(() => {
  backend();
});

afterEach(() => {
  setTransport(null);
  cleanup();
  useStore.getState().resetExport();
  resetRunSetup();
});

/** Finds the "KUKA handeye" preset card and returns its "Use preset" button. */
function presetUseButton(): HTMLElement {
  const card = screen.getByRole("article", { name: "KUKA handeye" });
  return within(card).getByRole("button", { name: "Use preset" });
}

describe("RunWorkspace happy path", () => {
  it("selects a preset, runs, and renders the mocked success response", async () => {
    render(
      <MemoryRouter>
        {/* main.tsx mounts the app under a TooltipProvider; ui's Tooltip needs it. */}
        <TooltipProvider>
          <RunWorkspace />
        </TooltipProvider>
      </MemoryRouter>,
    );

    fireEvent.click(presetUseButton());

    // Preset load resolves asynchronously (fetch_sample + read_manifest +
    // default_config); the quick-start grid collapses to the active-
    // preset bar once state settles.
    await screen.findByText("KUKA handeye", {}, { timeout: 2000 });
    const runButton = await screen.findByRole("button", { name: "Run" });
    expect(runButton.hasAttribute("disabled")).toBe(false);

    fireEvent.click(runButton);

    const banner = await screen.findByText("Solve completed", {}, { timeout: 2000 });
    expect(banner).toBeTruthy();
    expect(screen.getByText(/42 ms/)).toBeTruthy();
    expect(screen.getByText(/20\/20 usable views/)).toBeTruthy();

    // The success payload is threaded into the shared store, ready for
    // the /diagnose handoff.
    expect(useStore.getState().data).toEqual(
      RUN_SUCCESS_RESPONSE.kind === "ok" ? RUN_SUCCESS_RESPONSE.export : null,
    );
  });
});

describe("RunWorkspace progress + cancellation", () => {
  it("shows the stage checklist for a running solve and cancels it", async () => {
    // The run reports one stage and then waits until it is cancelled — so
    // the running state (and its Cancel button) is observable.
    backend(
      {},
      {
        run_calibration: async (_args, control) => {
          control.emit({ stage: "detect" });
          await control.cancelled;
          throw new StudioError({ kind: "cancelled", message: "cancelled" });
        },
      },
    );

    render(
      <MemoryRouter>
        {/* main.tsx mounts the app under a TooltipProvider; ui's Tooltip needs it. */}
        <TooltipProvider>
          <RunWorkspace />
        </TooltipProvider>
      </MemoryRouter>,
    );

    fireEvent.click(presetUseButton());
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

describe("RunWorkspace with a project open", () => {
  afterEach(() => useProject.setState({ project: null, error: null }));

  it("saves a successful run into the project and reopens it from disk", async () => {
    const saved: { name: string; record: unknown }[] = [];
    backend({
      save_project_run: (args) => {
        saved.push({ name: args.name, record: args.record });
        return { name: args.name, diagnostics: null };
      },
      load_export: () => ({
        export: RUN_SUCCESS_RESPONSE.kind === "ok" ? RUN_SUCCESS_RESPONSE.export : {},
        export_dir: "/p/runs/run-001",
      }),
    });
    useProject.setState({
      project: { dir: "/p", dataset_dir: "/p", manifest: {}, runs: [] },
    });

    render(
      <MemoryRouter>
        <TooltipProvider>
          <RunWorkspace />
        </TooltipProvider>
      </MemoryRouter>,
    );
    fireEvent.click(presetUseButton());
    await screen.findByText("KUKA handeye", {}, { timeout: 2000 });
    fireEvent.click(await screen.findByRole("button", { name: "Run" }));

    await screen.findByText(/saved as runs\/run-001/, {}, { timeout: 2000 });
    expect(saved).toHaveLength(1);
    expect(saved[0]!.record).toMatchObject({
      export: RUN_SUCCESS_RESPONSE.kind === "ok" ? RUN_SUCCESS_RESPONSE.export : null,
      config: DEFAULT_CONFIG,
      diagnostics: { topology: "planar_intrinsics", usable_views: 20, total_views: 20 },
    });
    // The viewers show the saved export, not an in-memory copy.
    expect(useStore.getState().exportPath).toBe("/p/runs/run-001/export.json");
    expect(useProject.getState().project?.runs.map((r) => r.name)).toEqual(["run-001"]);
  });
});
