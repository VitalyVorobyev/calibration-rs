/** The open project's unsaved edits: its dataset manifest (when the
 * project file holds it) and its workflow's configuration, as the Run
 * workspace's forms edit them. The project file on disk is the setup; this
 * holds only what is not saved yet, and `save` writes it through
 * `save_project` (the configuration as its changes from the library's
 * defaults, `config_patch`). */
import { create } from "zustand";
import { api } from "../api";
import type { DatasetSpec, ProblemKind, ProjectView } from "../api/generated/types";
import { inlineManifest, withManifest } from "./projectModel";
import { useProject } from "./projectStore";

/** The topology of a manifest value; `null` when it names none. */
export function topologyOf(manifest: unknown): ProblemKind | null {
  const m = manifest as Record<string, unknown> | null;
  return typeof m?.topology === "string" ? (m.topology as ProblemKind) : null;
}

export interface DraftState {
  /** `dir|revision` of the project version the draft starts from. */
  base: string | null;
  /** The manifest the forms show; edited only when `editable`. */
  manifest: DatasetSpec | null;
  /** Whether the project file holds the manifest (a referenced file is
   * never written). */
  editable: boolean;
  /** The workflow's configuration; `null` until it is known. */
  config: unknown;
  manifestDirty: boolean;
  configDirty: boolean;
  /** Start over from `view` (the project as last read), dropping edits. */
  resetFrom: (view: ProjectView | null) => void;
  setManifest: (next: DatasetSpec) => void;
  setConfig: (next: unknown) => void;
  /** Write the edits to the project; `true` when nothing is left unsaved. */
  save: () => Promise<boolean>;
}

/** Bumped by every configuration reset, so a slower default for an older
 * topology never overwrites a newer one's. */
let configRequest = 0;

const EMPTY = {
  base: null,
  manifest: null,
  editable: false,
  config: null,
  manifestDirty: false,
  configDirty: false,
};

export const useDraft = create<DraftState>()((set, get) => {
  /** The library's default configuration of `topology`. */
  const resetConfig = async (topology: ProblemKind | null) => {
    const request = ++configRequest;
    if (topology === null) {
      set({ config: null });
      return;
    }
    try {
      const form = await api.configForm({ topology });
      if (request === configRequest) set({ config: form.default });
    } catch {
      if (request === configRequest) set({ config: null });
    }
  };

  return {
    ...EMPTY,

    resetFrom: (view) => {
      configRequest++;
      if (view === null) {
        set(EMPTY);
        return;
      }
      const manifest = view.dataset.kind === "ok" ? view.dataset.spec : null;
      set({
        base: `${view.dir}|${view.revision}`,
        manifest,
        editable: inlineManifest(view.spec) !== null,
        config: view.config.kind === "ok" ? view.config.config : null,
        manifestDirty: false,
        configDirty: false,
      });
      // An invalid configuration starts the form from the defaults.
      if (view.config.kind !== "ok") void resetConfig(topologyOf(manifest));
    },

    setManifest: (next) => {
      const prev = get().manifest;
      set({ manifest: next, manifestDirty: true });
      // Another workflow has another configuration.
      if (topologyOf(next) !== topologyOf(prev)) {
        set({ configDirty: true });
        void resetConfig(topologyOf(next));
      }
    },

    setConfig: (config) => {
      configRequest++;
      set({ config, configDirty: true });
    },

    save: async () => {
      const { manifest, config, manifestDirty, configDirty, editable } = get();
      const project = useProject.getState().project;
      if (!project) return false;
      if (!manifestDirty && !configDirty) return true;
      let spec = project.spec;
      if (manifestDirty && editable && manifest !== null) {
        spec = withManifest(spec, manifest);
      }
      const topology = topologyOf(manifest);
      if (configDirty && topology !== null && config !== null) {
        try {
          spec = { ...spec, config: await api.configPatch({ topology, config }) };
        } catch (e) {
          useProject.setState({
            error: {
              kind: "config",
              message: e instanceof Error ? e.message : String(e),
            },
          });
          return false;
        }
      }
      // A saved project is re-read; the draft starts again from it.
      return useProject.getState().save(spec);
    },
  };
});

/** Whether the draft holds edits not saved to the project. */
export function isDirty(
  state: Pick<DraftState, "manifestDirty" | "configDirty">,
): boolean {
  return state.manifestDirty || state.configDirty;
}

// The draft follows the project: a newly opened or saved version (another
// revision) starts it again.
function follow(view: ProjectView | null): void {
  const base = view ? `${view.dir}|${view.revision}` : null;
  if (base !== useDraft.getState().base) useDraft.getState().resetFrom(view);
}
useProject.subscribe((state) => follow(state.project));
follow(useProject.getState().project);
