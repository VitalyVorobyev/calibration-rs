/** The configuration a calibration is about to run with: the Calibrate
 * screen's working copy of the project's configuration, what it is compared
 * with, and the run that starts from it.
 *
 * Nothing here writes the project. A run takes the working configuration as
 * a one-off patch over the project's (`calibrate`'s `config_patch`), and the
 * run records it; `saveAsDefault` is the one separate, explicit write of it
 * into the project file. The working copy is `null` until the first edit:
 * it then is the project's configuration, whatever that becomes. */
import type { JsonSchema } from "@vitavision/forms";
import { create } from "zustand";
import { api, asStudioError, isCancelled } from "../api";
import type {
  ProblemKind,
  ProjectView,
  RunInfo,
  StudioError,
} from "../api/generated/types";
import { advice, stageOf } from "./configCheck";
import { isEmptyPatch, patchOver } from "./configEdit";
import { useProject } from "./projectStore";
import { recordedPatch } from "./runRows";

/** What the editor's changes are measured from. */
export type Baseline =
  | { kind: "default" }
  | { kind: "project" }
  | { kind: "run"; id: string; config: unknown };

/** A calibration that did not produce a run. */
export interface RunFailure {
  error: StudioError;
  /** Where it stopped, in words; `null` when it sent no progress. */
  stage: string | null;
  /** What to try. */
  advice: string;
  /** The run it was meant to follow, if the editor came from one. */
  fromRun: string | null;
}

export interface EditorForm {
  topology: ProblemKind;
  schema: JsonSchema;
  /** The library's default configuration. */
  default: unknown;
}

export interface RunEditorState {
  /** The project the editor is for. */
  dir: string | null;
  form: EditorForm | null;
  formError: string | null;
  /** The edited configuration; `null` while it is the project's. */
  edits: unknown;
  baseline: Baseline;
  /** The run the editor was loaded from ("rerun from this run"). */
  fromRun: string | null;
  /** The editor's patch from the library defaults, when it is valid: what
   * a run's recorded patch is compared with. */
  editorPatch: Record<string, unknown> | null;
  starting: boolean;
  /** The job a run of this editor started, while it runs. */
  jobId: string | null;
  failure: RunFailure | null;
  /** A cancelled run says so once. */
  cancelled: boolean;

  /** Follow the open project: its folder and its workflow. */
  follow: (project: ProjectView | null) => void;
  edit: (next: unknown) => void;
  /** Put the editor back to the project's configuration. */
  reset: () => void;
  setBaseline: (kind: "default" | "project" | "run", runId?: string) => Promise<void>;
  /** Load a run's recorded configuration as the working copy. */
  loadRun: (run: RunInfo) => Promise<boolean>;
  setEditorPatch: (patch: Record<string, unknown> | null) => void;
  /** Write the working configuration to the project file as its default. */
  saveAsDefault: (project: ProjectView, working: unknown) => Promise<boolean>;
  /** Start a calibration on the working configuration. */
  run: (opts: { label: string; redetect: boolean }) => Promise<void>;
  clearFailure: () => void;
  clearCancelled: () => void;
}

/** The topology of a project's dataset; `null` while it does not read. */
export function projectTopology(project: ProjectView | null): ProblemKind | null {
  if (project?.dataset.kind !== "ok") return null;
  const topology = (project.dataset.spec as { topology?: unknown }).topology;
  return typeof topology === "string" ? (topology as ProblemKind) : null;
}

/** The project's configuration as a form value; the library defaults while
 * it does not resolve. */
export function projectConfig(
  project: ProjectView | null,
  form: EditorForm | null,
): unknown {
  if (project?.config.kind === "ok") return project.config.config;
  return form?.default ?? null;
}

/** The configuration the editor holds. */
export function workingConfig(
  state: RunEditorState,
  project: ProjectView | null,
): unknown {
  return state.edits ?? projectConfig(project, state.form);
}

/** What the editor compares with. */
export function baselineValue(
  baseline: Baseline,
  project: ProjectView | null,
  form: EditorForm | null,
): unknown {
  switch (baseline.kind) {
    case "default":
      return form?.default ?? null;
    case "project":
      return projectConfig(project, form);
    case "run":
      return baseline.config;
  }
}

/** The one-off patch over the project's configuration that a run takes. */
export function oneOffPatch(
  state: RunEditorState,
  project: ProjectView,
): Record<string, unknown> {
  if (state.edits === null) return {};
  return patchOver(state.edits, projectConfig(project, state.form));
}

/** Bumped by every `follow` that changes the workflow, so a slower form
 * for an earlier one never replaces a newer one's. */
let formRequest = 0;

const START = {
  dir: null,
  form: null,
  formError: null,
  edits: null,
  baseline: { kind: "default" } as Baseline,
  fromRun: null,
  editorPatch: null,
  starting: false,
  jobId: null,
  failure: null,
  cancelled: false,
};

export const useRunEditor = create<RunEditorState>()((set, get) => ({
  ...START,

  follow: (project) => {
    const dir = project?.dir ?? null;
    const topology = projectTopology(project);
    const state = get();
    if (dir !== state.dir) {
      formRequest++;
      set({ ...START, dir });
    }
    if (topology === null) {
      if (get().form !== null) set({ form: null });
      return;
    }
    if (get().form?.topology === topology) return;
    const request = ++formRequest;
    // Another workflow has another configuration: nothing carries over.
    set({
      form: null,
      formError: null,
      edits: null,
      fromRun: null,
      baseline: { kind: "default" },
      editorPatch: null,
    });
    api.configForm({ topology }).then(
      (form) => {
        if (request !== formRequest) return;
        if (!form?.schema) {
          set({ formError: "The back end sent no configuration form." });
          return;
        }
        set({
          form: { topology, schema: form.schema, default: form.default },
        });
      },
      (e: unknown) => {
        if (request === formRequest) set({ formError: asStudioError(e).message });
      },
    );
  },

  edit: (next) => set({ edits: next }),

  reset: () =>
    set({ edits: null, fromRun: null, baseline: { kind: "default" }, failure: null }),

  setBaseline: async (kind, runId) => {
    if (kind !== "run") {
      set({ baseline: { kind } });
      return;
    }
    const project = useProject.getState().project;
    const topology = projectTopology(project);
    const run = project?.runs.find((r) => r.id === runId);
    const patch = run ? recordedPatch(run) : null;
    if (!project || !topology || !run || !patch) return;
    try {
      const config = await api.resolveConfig({ topology, patch });
      set({ baseline: { kind: "run", id: run.id, config } });
    } catch (e) {
      useProject.setState({
        error: { kind: "config", message: asStudioError(e).message },
      });
    }
  },

  loadRun: async (run) => {
    const project = useProject.getState().project;
    const topology = projectTopology(project);
    const patch = recordedPatch(run);
    if (!project || !topology || !patch) return false;
    try {
      const config = await api.resolveConfig({ topology, patch });
      set({
        edits: config,
        baseline: { kind: "run", id: run.id, config },
        fromRun: run.id,
        failure: null,
      });
      return true;
    } catch (e) {
      useProject.setState({
        error: {
          kind: "config",
          message: `The configuration of ${run.id} does not resolve any more: ${asStudioError(e).message}`,
        },
      });
      return false;
    }
  },

  setEditorPatch: (editorPatch) => set({ editorPatch }),

  saveAsDefault: async (project, working) => {
    const topology = projectTopology(project);
    if (!topology) return false;
    try {
      const patch = await api.configPatch({ topology, config: working });
      const saved = await useProject.getState().save({ ...project.spec, config: patch });
      // The project now says what the editor says: no edits left to carry.
      if (saved) set({ edits: null, fromRun: null, baseline: { kind: "project" } });
      return saved;
    } catch (e) {
      useProject.setState({
        error: { kind: "config", message: asStudioError(e).message },
      });
      return false;
    }
  },

  run: async ({ label, redetect }) => {
    const state = get();
    const project = useProject.getState().project;
    if (!project || state.starting || state.jobId !== null) return;
    const patch = oneOffPatch(state, project);
    const fromRun = state.fromRun;
    set({ starting: true, failure: null, cancelled: false });
    let last: unknown = null;
    try {
      const job = await api.jobs.calibrate({
        dir: project.dir,
        ...(isEmptyPatch(patch) ? {} : { config_patch: patch }),
        ...(label.trim() ? { notes: { label: label.trim() } } : {}),
        ...(redetect ? { redetect: true } : {}),
      });
      set({ starting: false, jobId: job.id });
      job.onEvent((event) => {
        last = event;
      });
      const written = await job.result;
      set({ jobId: null, fromRun: null });
      await useProject.getState().refreshRuns();
      await useProject.getState().openRun(written.id);
    } catch (e) {
      set({ starting: false, jobId: null });
      if (isCancelled(e)) {
        set({ cancelled: true });
        return;
      }
      const error = asStudioError(e);
      set({
        failure: {
          error,
          stage: stageOf(last),
          advice: advice(error.kind, error.message),
          fromRun,
        },
      });
    }
  },

  clearFailure: () => set({ failure: null }),
  clearCancelled: () => set({ cancelled: false }),
}));

// The editor follows the open project.
useProject.subscribe((state) => useRunEditor.getState().follow(state.project));
useRunEditor.getState().follow(useProject.getState().project);
