// @vitest-environment jsdom
/** The project store against a mocked back end: opening adopts the
 * project and remembers it, a non-project is a typed error that drops off
 * the recent list, a save sends the revision it was opened at (and a stale
 * one is `changed_on_disk`), and runs are opened, noted, deleted and
 * imported by the project folder and their id. */
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { setTransport, StudioError } from "../api";
import {
  type CommandHandlers,
  createMockTransport,
  type MockTransport,
} from "../api/transport/mock";
import { useStore } from "../store";
import { calibratedRun, projectView } from "../test/projectFixtures";
import { useDraft } from "./draftStore";
import type { ProjectError } from "./projectModel";
import { useProject } from "./projectStore";

const RUN = calibratedRun("/p", "run-001");
const PROJECT = projectView("/p", [RUN]);
const EXPORT = { kind: "planar_intrinsics", diagnostics: {} };

let mock: MockTransport;

/** A command's rejection: the back end's error envelope. */
function rejection(error: ProjectError): Promise<never> {
  return Promise.reject(new StudioError(error));
}

function backend(commands: CommandHandlers) {
  mock = createMockTransport({
    commands: {
      config_form: () => ({ schema: {}, default: {} }),
      run_export: () => ({ export: EXPORT, image_root: "/p" }),
      ...commands,
    },
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
  useStore.getState().resetExport();
  useProject.setState({ project: null, error: null });
});

describe("useProject", () => {
  it("opens a project and remembers it", async () => {
    backend({ open_project: () => PROJECT, recent_projects: () => ["/p", "/older"] });
    expect(await useProject.getState().open("/p")).toBe(true);
    expect(useProject.getState().project).toEqual(PROJECT);
    expect(useProject.getState().error).toBeNull();
    await new Promise((r) => setTimeout(r, 0));
    expect(useProject.getState().recent).toEqual(["/p", "/older"]);
    // The draft starts from the project's manifest.
    expect(useDraft.getState().manifest).toEqual(PROJECT.spec.dataset.manifest);
  });

  it("reports a folder that is not a project and forgets it", async () => {
    backend({
      open_project: () =>
        rejection({ kind: "not_a_project", message: "/x is not a calibration project" }),
      forget_recent: () => [],
    });
    useProject.setState({ recent: ["/x"] });
    expect(await useProject.getState().open("/x")).toBe(false);
    expect(useProject.getState().error?.kind).toBe("not_a_project");
    expect(useProject.getState().recent).toEqual([]);
    expect(calls("forget_recent")).toHaveLength(1);
  });

  it("saves over the revision it opened, and reports a stale one", async () => {
    let revision = "sha256:r1";
    backend({
      open_project: () => PROJECT,
      save_project: (args) => {
        if (args.revision !== revision) {
          return rejection({ kind: "changed_on_disk", message: "changed on disk" });
        }
        revision = "sha256:r2";
        return { ...PROJECT, revision, spec: args.spec };
      },
    });
    await useProject.getState().open("/p");
    const named = { ...PROJECT.spec, name: "bench" };
    expect(await useProject.getState().save(named)).toBe(true);
    expect(useProject.getState().project?.revision).toBe("sha256:r2");
    expect(calls("save_project")[0]?.args).toEqual({
      dir: "/p",
      revision: "sha256:r1",
      spec: named,
    });
    // Another program saved meanwhile: the old revision is refused.
    useProject.setState({ project: PROJECT });
    expect(await useProject.getState().save(named)).toBe(false);
    expect(useProject.getState().error?.kind).toBe("changed_on_disk");
  });

  it("opens, notes and deletes a run by the project folder and its id", async () => {
    backend({
      open_project: () => PROJECT,
      set_run_notes: (args) => ({ ...RUN, notes: args.notes }),
      delete_run: () => [],
    });
    await useProject.getState().open("/p");
    await useProject.getState().openRun("run-001");
    expect(calls("run_export")[0]?.args).toEqual({ run: { dir: "/p", id: "run-001" } });
    expect(useStore.getState().run).toEqual({ dir: "/p", id: "run-001" });

    expect(await useProject.getState().setNotes("run-001", { label: "tight" })).toBe(
      true,
    );
    expect(useProject.getState().project?.runs[0]?.notes).toEqual({ label: "tight" });

    expect(await useProject.getState().deleteRun("run-001")).toBe(true);
    expect(useProject.getState().project?.runs).toEqual([]);
    // The deleted run is no longer shown.
    expect(useStore.getState().run).toBeNull();
  });

  it("imports an export as the next run and shows it", async () => {
    const imported = calibratedRun("/p", "run-002");
    backend({
      open_project: () => PROJECT,
      import_export: () => imported,
      list_runs: () => [RUN, imported],
    });
    await useProject.getState().open("/p");
    expect(await useProject.getState().importExport("/p/vcal/calibration.json")).toBe(
      true,
    );
    expect(calls("import_export")[0]?.args).toEqual({
      dir: "/p",
      export: "/p/vcal/calibration.json",
    });
    expect(useProject.getState().project?.runs.map((r) => r.id)).toEqual([
      "run-001",
      "run-002",
    ]);
    expect(useStore.getState().run).toEqual({ dir: "/p", id: "run-002" });
  });

  it("drops a run of another project from the viewers when one opens", async () => {
    backend({ open_project: (args) => projectView(args.dir) });
    useStore.getState().showRun(
      { dir: "/other", id: "run-001" },
      {
        export: EXPORT,
        image_root: null,
      },
    );
    await useProject.getState().open("/p");
    expect(useStore.getState().run).toBeNull();
  });
});
