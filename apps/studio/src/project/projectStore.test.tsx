// @vitest-environment jsdom
/** The project store against a mocked backend: opening loads the Run
 * setup, a non-project is a typed error that drops off the recent list,
 * creating uses the current setup, and runs are saved under the next free
 * name. */
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { setTransport, StudioError } from "../api";
import {
  type CommandHandlers,
  createMockTransport,
  type MockTransport,
} from "../api/transport/mock";
import { useStore } from "../store";
import { resetRunSetup, useRunSetup } from "../store/runSetup";
import type { ProjectError, ProjectInfo } from "./projectApi";
import { useProject } from "./projectStore";

const MANIFEST = { version: 1, topology: "planar_intrinsics", cameras: [] };
const CONFIG = { max_iters: 7 };

const PROJECT: ProjectInfo = {
  dir: "/p",
  dataset_dir: "/p/images",
  manifest: MANIFEST,
  runs: [{ name: "run-001", diagnostics: null }],
};

let mock: MockTransport;

/** A command's rejection: the back end's error envelope. */
function rejection(error: ProjectError): Promise<never> {
  return Promise.reject(new StudioError(error));
}

function backend(commands: CommandHandlers) {
  mock = createMockTransport({
    commands: { config_form: () => ({ schema: {}, default: CONFIG }), ...commands },
  });
  setTransport(mock);
}

function calls(command: string) {
  return mock.calls.filter((c) => c.command === command);
}

beforeEach(() => {
  useProject.setState({ project: null, recent: [], error: null });
});

afterEach(() => {
  setTransport(null);
  resetRunSetup();
  useStore.getState().resetExport();
});

describe("useProject", () => {
  it("opens a project into the Run setup and remembers it", async () => {
    // The back end keeps the recent projects; opening one puts it first.
    backend({ open_project: () => PROJECT, recent_projects: () => ["/p", "/older"] });
    expect(await useProject.getState().open("/p")).toBe(true);
    await new Promise((resolve) => setTimeout(resolve, 0));

    const project = useProject.getState();
    expect(project.project).toEqual(PROJECT);
    expect(project.recent).toEqual(["/p", "/older"]);
    expect(project.error).toBeNull();

    const setup = useRunSetup.getState();
    expect(setup.manifest).toEqual(MANIFEST);
    expect(setup.manifestDir).toBe("/p/images");
    expect(setup.manifestPath).toBe("/p/dataset.json");
    expect(setup.config).toEqual(CONFIG);
  });

  it("reports a folder that is not a project and forgets it", async () => {
    useProject.setState({ recent: ["/gone", "/p"] });
    backend({
      open_project: () =>
        rejection({
          kind: "not_a_project",
          message: "/gone is not a calibration project",
        }),
      forget_recent: ({ dir }) => ["/gone", "/p"].filter((d) => d !== dir),
    });
    expect(await useProject.getState().open("/gone")).toBe(false);

    const state = useProject.getState();
    expect(state.project).toBeNull();
    expect(state.error).toEqual({
      kind: "not_a_project",
      message: "/gone is not a calibration project",
    });
    expect(state.recent).toEqual(["/p"]);
  });

  it("creates a project from the current dataset", async () => {
    await useRunSetup.getState().loadSetup({
      manifest: MANIFEST,
      manifestDir: "/data/kuka_1",
      manifestPath: "/data/kuka_1/dataset.toml",
      presetId: "kuka",
    });
    backend({ create_project: () => ({ ...PROJECT, runs: [] }) });
    expect(await useProject.getState().create("/p")).toBe(true);

    expect(calls("create_project")[0]?.args).toEqual({
      dir: "/p",
      dataset_dir: "/data/kuka_1",
      manifest: MANIFEST,
    });
    expect(useProject.getState().project?.dir).toBe("/p");
  });

  it("refuses to create a project without a dataset", async () => {
    backend({});
    expect(await useProject.getState().create("/p")).toBe(false);
    expect(useProject.getState().error?.kind).toBe("io");
    expect(calls("create_project")).toEqual([]);
  });

  it("saves runs under the next free name and lists them", async () => {
    useProject.setState({ project: PROJECT });
    backend({
      save_project_run: (args) => ({
        name: args.name,
        diagnostics: args.record.diagnostics,
      }),
    });
    const diagnostics = {
      topology: "planar_intrinsics",
      usable_views: 3,
      total_views: 4,
      duration_ms: 10,
      cache_used: false,
    };
    const saved = await useProject.getState().saveRun({
      manifest: MANIFEST,
      config: CONFIG,
      export: {},
      dataset_dir: "/p",
      diagnostics,
    });
    expect(saved?.name).toBe("run-002");
    expect(useProject.getState().project?.runs.map((r) => r.name)).toEqual([
      "run-001",
      "run-002",
    ]);
  });

  it("keeps a run-saving failure as the project error", async () => {
    useProject.setState({ project: PROJECT });
    backend({
      save_project_run: () =>
        rejection({ kind: "run_exists", message: "run-002 exists" }),
    });
    const saved = await useProject.getState().saveRun({
      manifest: MANIFEST,
      config: CONFIG,
      export: {},
      dataset_dir: "/p",
      diagnostics: {
        topology: "planar_intrinsics",
        usable_views: 0,
        total_views: 0,
        duration_ms: 0,
        cache_used: false,
      },
    });
    expect(saved).toBeNull();
    expect(useProject.getState().error?.kind).toBe("run_exists");
  });

  it("saves the Run setup's manifest", async () => {
    useProject.setState({ project: PROJECT });
    const edited = { ...MANIFEST, description: "edited" };
    useRunSetup.getState().setManifest(edited);
    backend({ save_project_manifest: () => null });
    expect(await useProject.getState().saveManifest()).toBe(true);
    expect(calls("save_project_manifest")[0]?.args).toEqual({
      dir: "/p",
      manifest: edited,
    });
    expect(useProject.getState().project?.manifest).toEqual(edited);
  });

  it("opens a saved run's export", async () => {
    useProject.setState({ project: PROJECT });
    backend({
      load_export: () =>
        Promise.reject(new StudioError({ kind: "io", message: "not here" })),
    });
    await useProject.getState().openRun("run-001");
    expect(calls("load_export")[0]?.args).toEqual({
      path: "/p/runs/run-001/export.json",
    });
  });
});
