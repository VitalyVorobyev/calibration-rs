/** The open project, the recent-projects list and the last project error.
 *
 * Opening a project loads its manifest into the Run setup (the dataset
 * folder is the project's `dataset_dir`); saving writes the setup's
 * manifest back; a run saved into the project is reopened from its
 * `export.json`, so what Diagnose shows is what is on disk. */
import { create } from "zustand";
import { useStore } from "../store";
import { useRunSetup } from "../store/runSetup";
import {
  asProjectError,
  createProject,
  openProject,
  projectManifestPath,
  runExportPath,
  saveProjectManifest,
  saveProjectRun,
  type ProjectError,
  type ProjectInfo,
  type RunRecord,
  type RunSummary,
} from "./projectApi";
import {
  forgetRecent,
  loadRecent,
  nextRunName,
  rememberRecent,
  saveRecent,
} from "./projectModel";

export interface ProjectState {
  project: ProjectInfo | null;
  recent: string[];
  /** The last failed project command, shown by the shell until dismissed. */
  error: ProjectError | null;
  /** Open the project in `dir` and load its manifest into the Run setup. */
  open: (dir: string) => Promise<boolean>;
  /** Make `dir` a project for the current Run setup's dataset. */
  create: (dir: string) => Promise<boolean>;
  /** Write the Run setup's manifest to the project's `dataset.json`. */
  saveManifest: () => Promise<boolean>;
  /** Save a run under the next free `run-NNN` name. */
  saveRun: (record: RunRecord) => Promise<RunSummary | null>;
  /** Load a saved run's export into the viewer workspaces. */
  openRun: (name: string) => Promise<void>;
  close: () => void;
  clearError: () => void;
}

export const useProject = create<ProjectState>()((set, get) => {
  const fail = (e: unknown) => {
    set({ error: asProjectError(e) });
    return false;
  };

  const adopt = async (info: ProjectInfo) => {
    const recent = rememberRecent(get().recent, info.dir);
    saveRecent(recent);
    set({ project: info, recent, error: null });
    await useRunSetup.getState().loadSetup({
      manifest: info.manifest,
      manifestDir: info.dataset_dir,
      manifestPath: projectManifestPath(info.dir),
      presetId: null,
    });
  };

  return {
    project: null,
    recent: loadRecent(),
    error: null,

    open: async (dir) => {
      try {
        await adopt(await openProject(dir));
        return true;
      } catch (e) {
        const error = asProjectError(e);
        // A remembered folder that no longer opens drops off the list.
        if (error.kind === "not_a_project" || error.kind === "io") {
          const recent = forgetRecent(get().recent, dir);
          saveRecent(recent);
          set({ recent });
        }
        return fail(error);
      }
    },

    create: async (dir) => {
      const { manifest, manifestDir } = useRunSetup.getState();
      if (manifestDir === null) {
        return fail({
          kind: "io",
          message: "Choose a dataset first (a preset, a sniffed folder or a manifest).",
        });
      }
      try {
        await adopt(await createProject(dir, manifestDir, manifest));
        return true;
      } catch (e) {
        return fail(e);
      }
    },

    saveManifest: async () => {
      const project = get().project;
      if (!project) return false;
      const manifest = useRunSetup.getState().manifest;
      try {
        await saveProjectManifest(project.dir, manifest);
        // Stored as written; the form's value is a JSON object.
        set({
          project: { ...project, manifest: manifest as ProjectInfo["manifest"] },
          error: null,
        });
        return true;
      } catch (e) {
        return fail(e);
      }
    },

    saveRun: async (record) => {
      const project = get().project;
      if (!project) return null;
      const name = nextRunName(project.runs.map((r) => r.name));
      try {
        const summary = await saveProjectRun(project.dir, name, record);
        // Re-read the project from the store: it may have changed while saving.
        const current = get().project;
        if (current?.dir === project.dir) {
          const runs = [...current.runs, summary].sort((a, b) =>
            a.name.localeCompare(b.name),
          );
          set({ project: { ...current, runs }, error: null });
        }
        return summary;
      } catch (e) {
        fail(e);
        return null;
      }
    },

    openRun: async (name) => {
      const project = get().project;
      if (!project) return;
      await useStore.getState().loadExport(runExportPath(project.dir, name));
    },

    close: () => set({ project: null, error: null }),

    clearError: () => set({ error: null }),
  };
});
