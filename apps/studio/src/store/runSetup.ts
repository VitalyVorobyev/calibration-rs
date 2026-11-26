/** The Run workspace's setup — the dataset manifest, the calibration config
 * and where the dataset lives — as an app-wide store. It outlives the Run
 * workspace's mount, so opening a project from the shell fills it while
 * another workspace is showing, and a saved run records exactly what ran.
 *
 * Every default comes from the back end (`dataset_form`, `config_form`,
 * `resolve_config`: the library's own defaults and merge rule); the store
 * starts empty (`null`) until `ensureDefaults` or a loaded setup fills it. */
import { create } from "zustand";
import { api } from "../api";
import type { ProblemKind } from "../api/generated/types";

/** Topology of a manifest value, defaulting to planar. */
export function topologyOf(manifest: unknown): string {
  const m = manifest as Record<string, unknown> | null;
  return typeof m?.topology === "string" ? m.topology : "planar_intrinsics";
}

/** The config of `topology` that the library makes from its defaults and
 * `overrides` (none: the defaults); `null` when the back end cannot say
 * (an unknown topology, no back end). */
export async function fetchDefaultConfig(
  topology: string,
  overrides?: Record<string, unknown>,
): Promise<unknown> {
  const kind = topology as ProblemKind;
  try {
    if (overrides !== undefined) {
      return await api.resolveConfig({ topology: kind, patch: overrides });
    }
    return (await api.configForm({ topology: kind })).default;
  } catch {
    return null;
  }
}

/** A whole new setup, as a preset, a sniffed folder or a project gives it. */
export interface SetupSource {
  manifest: unknown;
  /** The folder the manifest's relative paths resolve against. */
  manifestDir: string;
  /** The manifest file, when there is one on disk. */
  manifestPath: string | null;
  /** The Quick Start preset it came from, if any. */
  presetId: string | null;
  /** Applied over the topology's default config by the library's rule (a
   * preset's tuning). */
  configOverrides?: Record<string, unknown> | undefined;
}

export interface RunSetupState {
  /** The manifest; `null` until the defaults have loaded. */
  manifest: unknown;
  /** The topology's config; `null` until it has loaded (or when it could
   * not be). */
  config: unknown;
  manifestDir: string | null;
  manifestPath: string | null;
  presetId: string | null;
  /** Replace the manifest. A new topology resets the config to that
   * topology's defaults (the config schema differs per topology). */
  setManifest: (next: unknown) => void;
  /** `setManifest` with a function of the current manifest. */
  updateManifest: (update: (prev: unknown) => unknown) => void;
  setConfig: (next: unknown) => void;
  /** Point at another dataset folder or manifest file; leaves any preset. */
  setPaths: (paths: { manifestDir: string; manifestPath?: string | null }) => void;
  /** Start over from `source`, with that topology's default config. */
  loadSetup: (source: SetupSource) => Promise<void>;
  /** Fill an empty setup with the back end's default manifest and its
   * topology's default config; a setup already there is kept. */
  ensureDefaults: () => Promise<void>;
}

const INITIAL = {
  manifest: null,
  config: null,
  manifestDir: null,
  manifestPath: null,
  presetId: null,
};

/** Bumped by every config reset, so a slower defaults fetch for an older
 * manifest never overwrites a newer one's. */
let configRequest = 0;

export const useRunSetup = create<RunSetupState>()((set, get) => {
  const resetConfig = async (manifest: unknown, overrides?: Record<string, unknown>) => {
    const request = ++configRequest;
    const config = await fetchDefaultConfig(topologyOf(manifest), overrides);
    if (request === configRequest) set({ config });
  };

  return {
    ...INITIAL,

    setManifest: (manifest) => {
      const prev = get().manifest;
      set({ manifest });
      if (topologyOf(manifest) !== topologyOf(prev) || get().config === null) {
        void resetConfig(manifest);
      }
    },

    updateManifest: (update) => get().setManifest(update(get().manifest)),

    setConfig: (config) => {
      configRequest++;
      set({ config });
    },

    setPaths: ({ manifestDir, manifestPath }) =>
      set((s) => ({
        manifestDir,
        manifestPath: manifestPath === undefined ? s.manifestPath : manifestPath,
        presetId: null,
      })),

    loadSetup: async (source) => {
      set({
        manifest: source.manifest,
        manifestDir: source.manifestDir,
        manifestPath: source.manifestPath,
        presetId: source.presetId,
      });
      await resetConfig(source.manifest, source.configOverrides);
    },

    ensureDefaults: async () => {
      if (get().manifest !== null) {
        if (get().config === null) await resetConfig(get().manifest);
        return;
      }
      let manifest: unknown;
      try {
        manifest = (await api.datasetForm()).default;
      } catch {
        return;
      }
      // A setup loaded meanwhile wins.
      if (get().manifest !== null) return;
      set({ manifest });
      await resetConfig(manifest);
    },
  };
});

/** Back to a fresh session (tests, closing a project). */
export function resetRunSetup(): void {
  configRequest++;
  useRunSetup.setState(INITIAL);
}
