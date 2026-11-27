// @vitest-environment jsdom
/** The Run workspace against a mock transport: with no project it offers
 * the ways to one (a Quick Start sample becomes a project); with one open,
 * the forms show its manifest and configuration, an edit is saved before
 * the run, and the `calibrate` job's run is listed and shown; a running job
 * shows its stages and cancels; a choice the manifest leaves open asks. */
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { TooltipProvider } from "@vitavision/ui";
import { MemoryRouter } from "react-router";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { setTransport, StudioError } from "../../api";
import type { ProjectView } from "../../api/generated/types";
import {
  type CommandHandlers,
  createMockTransport,
  type JobHandlers,
  type MockTransport,
} from "../../api/transport/mock";
import { useDraft } from "../../project/draftStore";
import { useProject } from "../../project/projectStore";
import { useSession } from "../../session/session";
import { useStore } from "../../store";
import { calibratedRun, MANIFEST, projectView } from "../../test/projectFixtures";
import { RunWorkspace } from "./index";
import datasetSchema from "../../../../../schemas/dataset_spec.json";
import planarSchema from "../../../../../schemas/planar_intrinsics_config.json";

const DEFAULT_CONFIG = {
  intrinsics_init: { iterations: 2, fix_k3: true },
  solver: { max_iters: 50, verbosity: 0, backend: "tiny_solver" },
  camera: {
    model: { projection: "pinhole", distortion: "brown_conrady5", sensor: "identity" },
    fix: {},
  },
  optimize: { reprojection_loss: { kind: "none" }, fix_poses: [] },
};

const EXPORT = {
  kind: "planar_intrinsics",
  diagnostics: {
    per_feature_residuals: { target: [] },
    image_manifest: { root: ".", frames: [{ view: 0, camera: 0, path: "frame0.png" }] },
  },
};

/** The project `/p` as the back end answers it, with `runs`. */
function view(runs = projectView("/p").runs): ProjectView {
  return {
    ...projectView("/p", runs),
    config: { kind: "ok", config: DEFAULT_CONFIG },
  };
}

function backend(commands: CommandHandlers = {}, jobs: JobHandlers = {}): MockTransport {
  const mock = createMockTransport({
    commands: {
      config_form: () => ({ schema: planarSchema, default: DEFAULT_CONFIG }),
      dataset_form: () => ({ schema: datasetSchema, default: MANIFEST }),
      open_project: () => view(),
      recent_projects: () => [],
      list_runs: () => [calibratedRun("/p", "run-001", [20, 20])],
      run_export: () => ({ export: EXPORT, image_root: "/p" }),
      ...commands,
    },
    jobs: {
      calibrate: () => calibratedRun("/p", "run-001", [20, 20]),
      ...jobs,
    },
  });
  setTransport(mock);
  return mock;
}

function renderRun() {
  return render(
    <MemoryRouter>
      {/* main.tsx mounts the app under a TooltipProvider; ui's Tooltip needs it. */}
      <TooltipProvider>
        <RunWorkspace />
      </TooltipProvider>
    </MemoryRouter>,
  );
}

beforeEach(() => {
  backend();
});

afterEach(() => {
  setTransport(null);
  cleanup();
  useStore.getState().resetExport();
  useProject.setState({ project: null, error: null });
  useSession.setState({ status: { kind: "connecting" } });
});

describe("RunWorkspace without a project", () => {
  it("offers the ways to a project, and makes a Quick Start sample one", async () => {
    const mock = backend({}, { fetch_sample: () => ({ dir: "/samples/kuka_1" }) });
    useSession.setState({
      status: {
        kind: "ready",
        info: {
          version: "0",
          shell: "server",
          roots: [],
          roots_editable: false,
          samples: [
            {
              id: "kuka_1",
              description: "single-camera hand-eye",
              topology: "single_cam_handeye",
              licence: "MIT",
            },
          ],
        },
      },
    });
    renderRun();
    expect(screen.getByRole("button", { name: "Open project…" })).toBeTruthy();
    expect(
      screen.getByRole("button", { name: "New project from a dataset folder…" }),
    ).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Use the kuka_1 sample" }));
    await waitFor(() => expect(useProject.getState().project?.dir).toBe("/p"));
    expect(mock.calls.map((c) => c.command)).toEqual(
      expect.arrayContaining(["fetch_sample", "open_project"]),
    );
    expect(mock.calls.find((c) => c.command === "open_project")?.args).toEqual({
      dir: "/samples/kuka_1",
    });
  });
});

describe("RunWorkspace with a project open", () => {
  it("shows the project's configuration in the form the back end serves", async () => {
    useProject.setState({ project: view() });
    renderRun();
    fireEvent.click(screen.getByText("Calibration config"));
    const solver = await screen.findByRole("group", { name: "Solver" });
    const maxIters = within(solver).getByRole<HTMLInputElement>("spinbutton", {
      name: "Max iters",
    });
    expect(maxIters.value).toBe("50");
  });

  it("saves an edit, runs the project, lists the run and shows it", async () => {
    const saved: unknown[] = [];
    const mock = backend({
      config_patch: () => ({ solver: { max_iters: 7 } }),
      save_project: (args) => {
        saved.push(args);
        return { ...view(), revision: "sha256:r2", spec: args.spec };
      },
    });
    useProject.setState({ project: view() });
    renderRun();
    useDraft.getState().setConfig({ ...DEFAULT_CONFIG, solver: { max_iters: 7 } });
    fireEvent.click(await screen.findByRole("button", { name: "Run" }));

    await screen.findByText("Solve completed", {}, { timeout: 2000 });
    expect(
      screen.getByText(/42 ms · 20\/20 usable views · saved as runs\/run-001/),
    ).toBeTruthy();
    // Saved first, over the revision the project was opened at.
    expect(saved).toHaveLength(1);
    expect(saved[0]).toMatchObject({
      dir: "/p",
      revision: "sha256:r1",
      spec: { config: { solver: { max_iters: 7 } } },
    });
    const order = mock.calls.map((c) => c.command);
    expect(order.indexOf("save_project")).toBeLessThan(order.indexOf("calibrate"));
    expect(mock.calls.find((c) => c.command === "calibrate")?.args).toEqual({
      dir: "/p",
    });
    // The run is listed and shown, named by the project and its id.
    expect(useProject.getState().project?.runs.map((r) => r.id)).toEqual(["run-001"]);
    expect(useStore.getState().run).toEqual({ dir: "/p", id: "run-001" });
    expect(useStore.getState().data).toEqual(EXPORT);
  });

  it("shows the stage checklist for a running job and cancels it", async () => {
    backend(
      {},
      {
        calibrate: async (_args, control) => {
          control.emit({ kind: "detection", done: 1, total: 4 });
          await control.cancelled;
          throw new StudioError({ kind: "cancelled", message: "cancelled" });
        },
      },
    );
    useProject.setState({ project: view() });
    renderRun();
    fireEvent.click(await screen.findByRole("button", { name: "Run" }));
    const cancelButton = await screen.findByRole("button", { name: "Cancel" });
    expect(screen.getByText("Detecting features")).toBeTruthy();
    expect(await screen.findByText("1/4 images")).toBeTruthy();
    fireEvent.click(cancelButton);
    await screen.findByText("Run cancelled", {}, { timeout: 2000 });
    expect(useStore.getState().data).toBeNull();
  });

  it("asks for a choice the manifest leaves open, and applies it", async () => {
    backend(
      {},
      {
        calibrate: () => {
          throw new StudioError({
            kind: "needs_input",
            message: "How do the images pair into views?",
            field: "pose_pairing",
            suggestions: ["by_index"],
          });
        },
      },
    );
    useProject.setState({ project: view() });
    renderRun();
    fireEvent.click(await screen.findByRole("button", { name: "Run" }));
    await screen.findByText("How do the images pair into views?");
    fireEvent.click(screen.getByRole("button", { name: "by_index" }));
    expect(useDraft.getState().manifest).toMatchObject({
      pose_pairing: { kind: "by_index" },
    });
    expect(useDraft.getState().manifestDirty).toBe(true);
  });
});
