/** The open project (the back end's `ProjectView`, re-read after every
 * change), the recent-projects list (the back end's) and the last project
 * error.
 *
 * The back end keeps nothing open: every command names the project by its
 * folder, a save sends back the revision the project was opened at (a
 * newer file on disk is `changed_on_disk`), and a run is named by the
 * project folder and its id. Choosing a run loads its export into the
 * viewers (`useStore.loadRun`). */
import { create } from "zustand";
import { api } from "../api";
import type { ProjectSpec, ProjectView, RunInfo, RunNotes } from "../api/generated/types";
import { useStore } from "../store";
import { asProjectError, type ProjectError } from "./projectModel";

export type { ProjectError } from "./projectModel";

export interface ProjectState {
  project: ProjectView | null;
  recent: string[];
  /** The last failed project command, shown by the shell until dismissed. */
  error: ProjectError | null;
  /** Read the recent projects from the back end. */
  refreshRecent: () => Promise<void>;
  /** Open the project in `dir`. */
  open: (dir: string) => Promise<boolean>;
  /** Read the open project again (another tab or program changed it). */
  reload: () => Promise<boolean>;
  /** Make `dir` a project of `spec` and open it. */
  create: (dir: string, spec: ProjectSpec) => Promise<boolean>;
  /** Write `spec` to the open project, over the revision it was opened at. */
  save: (spec: ProjectSpec) => Promise<boolean>;
  /** Read the open project's runs again. */
  refreshRuns: () => Promise<void>;
  /** Load run `id` of the open project into the viewers. */
  openRun: (id: string) => Promise<void>;
  /** Replace the notes of run `id`. */
  setNotes: (id: string, notes: RunNotes) => Promise<boolean>;
  /** Change some of run `id`'s notes, on top of its latest ones: edits
   * made together (a star and a label) each keep the other's. */
  patchNotes: (id: string, patch: Partial<RunNotes>) => Promise<boolean>;
  /** Delete run `id`. */
  deleteRun: (id: string) => Promise<boolean>;
  /** Add the export at `path` as the next run, and show it. */
  importExport: (path: string) => Promise<boolean>;
  close: () => void;
  clearError: () => void;
}

/** Bumped by every open, so a slower answer for an earlier project never
 * replaces a newer one. */
let openSeq = 0;

/** The notes writes in flight, in order (`patchNotes`). */
let notesQueue: Promise<void> = Promise.resolve();

export const useProject = create<ProjectState>()((set, get) => {
  const fail = (e: unknown) => {
    set({ error: asProjectError(e) });
    return false;
  };

  const refreshRecent = async () => {
    try {
      set({ recent: await api.recentProjects() });
    } catch {
      // No back end (yet): the list stays as it is.
    }
  };

  /** Show `view` as the open project; the viewers drop a run of another. */
  const adopt = (view: ProjectView) => {
    set({ project: view, error: null });
    const shown = useStore.getState().run;
    if (shown !== null && shown.dir !== view.dir) useStore.getState().resetExport();
  };

  /** Replace the open project's runs (when it is still `dir`). */
  const setRuns = (dir: string, runs: RunInfo[]) => {
    const current = get().project;
    if (current?.dir === dir) set({ project: { ...current, runs } });
    const shown = useStore.getState().run;
    if (shown?.dir === dir && !runs.some((r) => r.id === shown.id)) {
      useStore.getState().resetExport();
    }
  };

  return {
    project: null,
    recent: [],
    error: null,
    refreshRecent,

    open: async (dir) => {
      const seq = ++openSeq;
      try {
        const view = await api.openProject({ dir });
        if (seq !== openSeq) return false;
        adopt(view);
        // The back end put it first among the recent projects.
        void refreshRecent();
        return true;
      } catch (e) {
        const error = asProjectError(e);
        // A remembered folder that no longer opens drops off the list.
        if (
          get().recent.includes(dir) &&
          (error.kind === "not_a_project" ||
            error.kind === "io" ||
            error.kind === "forbidden_path")
        ) {
          try {
            set({ recent: await api.forgetRecent({ dir }) });
          } catch {
            // Kept; it fails again next time.
          }
        }
        return fail(error);
      }
    },

    reload: async () => {
      const project = get().project;
      if (!project) return false;
      return get().open(project.dir);
    },

    create: async (dir, spec) => {
      try {
        adopt(await api.createProject({ dir, spec }));
        void refreshRecent();
        return true;
      } catch (e) {
        return fail(e);
      }
    },

    save: async (spec) => {
      const project = get().project;
      if (!project) return false;
      try {
        adopt(
          await api.saveProject({ dir: project.dir, revision: project.revision, spec }),
        );
        return true;
      } catch (e) {
        return fail(e);
      }
    },

    refreshRuns: async () => {
      const project = get().project;
      if (!project) return;
      try {
        setRuns(project.dir, await api.listRuns({ dir: project.dir }));
      } catch (e) {
        fail(e);
      }
    },

    openRun: async (id) => {
      const project = get().project;
      if (!project) return;
      await useStore.getState().loadRun({ dir: project.dir, id });
    },

    setNotes: async (id, notes) => {
      const project = get().project;
      if (!project) return false;
      try {
        const run = await api.setRunNotes({ dir: project.dir, id, notes });
        // The runs as they are now, not as they were when this was asked.
        const now = get().project;
        if (now?.dir === project.dir) {
          setRuns(
            project.dir,
            now.runs.map((r) => (r.id === id ? run : r)),
          );
        }
        return true;
      } catch (e) {
        return fail(e);
      }
    },

    patchNotes: (id, patch) => {
      // One at a time, each on the notes the one before left.
      const next = notesQueue.then(async () => {
        const current = get().project?.runs.find((r) => r.id === id)?.notes ?? {};
        return get().setNotes(id, { ...current, ...patch });
      });
      notesQueue = next.then(
        () => undefined,
        () => undefined,
      );
      return next;
    },

    deleteRun: async (id) => {
      const project = get().project;
      if (!project) return false;
      try {
        setRuns(project.dir, await api.deleteRun({ dir: project.dir, id }));
        return true;
      } catch (e) {
        return fail(e);
      }
    },

    importExport: async (path) => {
      const project = get().project;
      if (!project) return false;
      try {
        const run = await api.importExport({ dir: project.dir, export: path });
        await get().refreshRuns();
        await get().openRun(run.id);
        return true;
      } catch (e) {
        return fail(e);
      }
    },

    close: () => {
      openSeq++;
      set({ project: null, error: null });
      useStore.getState().resetExport();
    },

    clearError: () => set({ error: null }),
  };
});
