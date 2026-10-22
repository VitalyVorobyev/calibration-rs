/** The Run workspace's setup — the dataset manifest, the calibration config
 * and where the dataset lives — as an app-wide store. It outlives the Run
 * workspace's mount, so opening a project from the shell fills it while
 * another workspace is showing, and a saved run records exactly what ran. */
import { invoke } from "@tauri-apps/api/core";
import { create } from "zustand";
import { isTauriContext } from "../lib/tauri";
import planarDefaultConfig from "../schemas/planar_intrinsics_config.default.json";
import { mergeConfig } from "../workspaces/RunWorkspace/presets";

/** A fresh session's manifest: one chessboard camera, planar intrinsics. */
export const DEFAULT_DATASET: unknown = {
  version: 1,
  cameras: [
    {
      id: "cam0",
      images: { kind: "glob", pattern: "*.png" },
    },
  ],
  target: {
    kind: "chessboard",
    rows: 9,
    cols: 6,
    square_size_m: 0.025,
  },
  topology: "planar_intrinsics",
};

// Browser-context fallback only — inside Tauri the defaults come from
// `default_config_cmd` (Rust `Config::default()`), the single source of
// truth. This is the same value, emitted by `cargo xtask emit-schemas` (and
// drift-checked in CI), so it cannot fall behind the Rust default.
const DEFAULT_PLANAR_CONFIG: unknown = planarDefaultConfig;

/** Topology of a manifest value, defaulting to planar. */
export function topologyOf(manifest: unknown): string {
  const m = manifest as Record<string, unknown> | null;
  return typeof m?.topology === "string" ? m.topology : "planar_intrinsics";
}

/** Default config for a topology: Rust-side defaults inside Tauri,
 * a static fallback in plain-browser dev. */
export async function fetchDefaultConfig(topology: string): Promise<unknown> {
  if (isTauriContext()) {
    try {
      return await invoke<unknown>("default_config_cmd", { topology });
    } catch {
      // Fall through to the static fallback (e.g. unknown topology).
    }
  }
  return topology === "planar_intrinsics" ? DEFAULT_PLANAR_CONFIG : {};
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
  /** Merged over the topology's default config (a preset's tuning). */
  configOverrides?: Record<string, unknown> | undefined;
}

export interface RunSetupState {
  manifest: unknown;
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
}

const INITIAL = {
  manifest: DEFAULT_DATASET,
  config: DEFAULT_PLANAR_CONFIG,
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
    let config = await fetchDefaultConfig(topologyOf(manifest));
    if (overrides !== undefined) config = mergeConfig(config, overrides);
    if (request === configRequest) set({ config });
  };

  return {
    ...INITIAL,

    setManifest: (manifest) => {
      const prev = get().manifest;
      set({ manifest });
      if (topologyOf(manifest) !== topologyOf(prev)) void resetConfig(manifest);
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
  };
});

/** Back to a fresh session (tests, closing a project). */
export function resetRunSetup(): void {
  configRequest++;
  useRunSetup.setState(INITIAL);
}
