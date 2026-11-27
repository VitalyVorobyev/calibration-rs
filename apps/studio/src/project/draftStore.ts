/** The open project's unsaved edits: its dataset manifest (when the
 * project file holds it) and its workflow's configuration, as the Run
 * workspace's forms edit them. The project file on disk is the setup; this
 * holds only what is not saved yet, and `save` writes it through
 * `save_project` (the configuration as its changes from the library's
 * defaults, `config_patch`). */
import { diffPaths, getAtPath, setAtPath } from "@vitavision/forms";
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
  /** The manifest and configuration as saved at `base`: what the forms mark
   * "changed" against. */
  baseManifest: DatasetSpec | null;
  baseConfig: unknown;
  /** The project as the disk now has it when it changed under unsaved edits
   * (another tab or program saved it); `null` while the draft is on the
   * version it started from. The edits stay until `rebase` or
   * `keepEditing` says what becomes of them. */
  disk: ProjectView | null;
  /** A save is in flight: its own change of the file is not another's. */
  saving: boolean;
  /** Start over from `view` (the project as last read), dropping edits. */
  resetFrom: (view: ProjectView | null) => void;
  /** Drop the unsaved edits: start over from the project as last read. */
  revert: () => void;
  /** Take the disk's version as the base and put my edits back on it: a
   * field I changed keeps my value (marked changed, with the disk's as its
   * baseline); every other field takes the disk's. */
  rebase: () => void;
  /** Dismiss the disk's change: the next save overwrites it. */
  keepEditing: () => void;
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
  baseManifest: null,
  baseConfig: null,
  disk: null,
  saving: false,
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
      const config = view.config.kind === "ok" ? view.config.config : null;
      set({
        base: `${view.dir}|${view.revision}`,
        manifest,
        editable: inlineManifest(view.spec) !== null,
        config,
        manifestDirty: false,
        configDirty: false,
        baseManifest: manifest,
        baseConfig: config,
        disk: null,
        saving: false,
      });
      // An invalid configuration starts the form from the defaults.
      if (view.config.kind !== "ok") void resetConfig(topologyOf(manifest));
    },

    revert: () => get().resetFrom(useProject.getState().project),

    rebase: () => {
      const { disk, manifest, baseManifest, config, configDirty } = get();
      if (disk === null) return;
      const onDisk = disk.dataset.kind === "ok" ? disk.dataset.spec : null;
      const diskConfig = disk.config.kind === "ok" ? disk.config.config : null;
      let next = onDisk;
      if (onDisk !== null && manifest !== null && baseManifest !== null) {
        next = diffPaths(manifest, baseManifest).reduce(
          (acc, path) => setAtPath(acc, path, getAtPath(manifest, path)) as DatasetSpec,
          onDisk,
        );
      }
      set({
        base: `${disk.dir}|${disk.revision}`,
        manifest: next,
        baseManifest: onDisk,
        editable: inlineManifest(disk.spec) !== null,
        manifestDirty:
          next !== null && onDisk !== null && diffPaths(next, onDisk).length > 0,
        config: configDirty ? config : diskConfig,
        baseConfig: diskConfig,
        disk: null,
      });
    },

    keepEditing: () => {
      const { disk } = get();
      if (disk !== null) set({ base: `${disk.dir}|${disk.revision}`, disk: null });
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
      set({ saving: true });
      try {
        return await useProject.getState().save(spec);
      } finally {
        set({ saving: false });
      }
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
  const draft = useDraft.getState();
  if (base === draft.base) return;
  // Unsaved edits are not thrown away by a change of the project file
  // (another tab, or a save of the project's other fields): when the
  // manifest and configuration are what the draft started from, only the
  // base moves; when they changed too, the draft says so (`disk`) and waits.
  if (
    view !== null &&
    isDirty(draft) &&
    !draft.saving &&
    draft.base?.startsWith(`${view.dir}|`)
  ) {
    const onDisk = view.dataset.kind === "ok" ? view.dataset.spec : null;
    const diskConfig = view.config.kind === "ok" ? view.config.config : null;
    const same =
      diffPaths(onDisk, draft.baseManifest).length === 0 &&
      diffPaths(diskConfig, draft.baseConfig).length === 0;
    useDraft.setState(same ? { base, disk: null } : { disk: view });
    return;
  }
  draft.resetFrom(view);
}
useProject.subscribe((state) => follow(state.project));
follow(useProject.getState().project);
