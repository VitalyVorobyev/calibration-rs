/** Built-in preset registry for the Run workspace.
 *
 * Each enabled preset carries the *repo-root-relative* path to its TOML
 * manifest and enough metadata to render a meaningful card without
 * reading the file. `RunWorkspace` resolves the absolute path at
 * load time by joining `manifestPath` onto the workspace repo root
 * (`lib/tauri.ts`'s `repoRoot()`, backed by the `repo_root_cmd` Tauri
 * command — see `app/README.md`'s dev notes). This keeps the preset
 * registry free of any developer's personal absolute path while still
 * working out of the box for `bun run tauri dev` on any checkout.
 */

export interface EnabledPreset {
  id: string;
  disabled?: false;
  /** Human-readable name shown on the card. */
  name: string;
  /** Subtitle line: dataset group + camera label. */
  group: string;
  topology: string;
  targetKind: string;
  /** Short target description shown in the card body. */
  targetSummary: string;
  imageCount: number;
  /** Path to the TOML manifest file, relative to the repo root
   * (resolved to an absolute path via `repoRoot()` at load time). */
  manifestPath: string;
  /**
   * Optional config patch deep-merged over the topology's
   * `default_config_cmd` defaults when the preset is applied — for
   * datasets whose defaults are wrong (e.g. a rig of tilted-sensor
   * cameras observed EyeToHand). Keys mirror the Rust `*Config` JSON.
   */
  configOverrides?: Record<string, unknown>;
  /**
   * Optional manifest patch deep-merged over the loaded TOML manifest
   * before the run. Used for detector options that belong to
   * `DatasetSpec`, not the calibration config.
   */
  manifestOverrides?: Record<string, unknown>;
}

/** Recursively merge `patch` over `base` (objects only; arrays and
 *  scalars are replaced). Used to apply preset config overrides. */
export function mergeConfig(base: unknown, patch: Record<string, unknown>): unknown {
  if (typeof base !== "object" || base === null || Array.isArray(base)) {
    return patch;
  }
  const out: Record<string, unknown> = { ...(base as Record<string, unknown>) };
  for (const [key, value] of Object.entries(patch)) {
    if (
      typeof value === "object" &&
      value !== null &&
      !Array.isArray(value) &&
      typeof out[key] === "object" &&
      out[key] !== null &&
      !Array.isArray(out[key])
    ) {
      out[key] = mergeConfig(out[key], value as Record<string, unknown>);
    } else {
      out[key] = value;
    }
  }
  return out;
}

export interface DisabledPreset {
  id: string;
  disabled: true;
  name: string;
  group: string;
  topology: string;
  targetKind: string;
  targetSummary: string;
  imageCount: number | null;
  /** One-line explanation shown on the card. */
  disabledReason: string;
  /** Short badge text for the planned capability. */
  milestone: string;
}

export type Preset = EnabledPreset | DisabledPreset;

export const BUILTIN_PRESETS: Preset[] = [
  // ── Enabled: KUKA handeye (committed dataset) ────────────────────────────
  {
    id: "kuka-handeye",
    name: "KUKA handeye",
    group: "kuka_1 dataset",
    topology: "SingleCamHandeye",
    targetKind: "chessboard",
    targetSummary: "chessboard 17×28, 20 mm · robot poses 4×4",
    imageCount: 30,
    manifestPath: "data/kuka_1/dataset.toml",
  },

  // ── Enabled: DS8 rig hand-eye (public download) ──────────────────────
  {
    id: "ds8-rig-handeye",
    name: "DS8 rig hand-eye",
    group: "DS8 dataset (USDA, public domain)",
    topology: "RigHandeye",
    targetKind: "chessboard",
    targetSummary: "chessboard 10×14, 52 mm · 3 cameras · robot poses",
    imageCount: 42,
    manifestPath: "data/DS8/dataset.toml",
  },
];

/** Presets for datasets outside the repository, from an optional untracked
 *  `presets.local.ts` next to this file that exports
 *  `LOCAL_PRESETS: Preset[]`. Without that file this is empty. */
const localPresetModules = import.meta.glob<{ LOCAL_PRESETS: Preset[] }>(
  "./presets.local.ts",
  {
    eager: true,
  },
);

/** Every preset the Run workspace offers: the built-in ones, then the local ones. */
export const PRESETS: Preset[] = [
  ...BUILTIN_PRESETS,
  ...Object.values(localPresetModules).flatMap((m) => m.LOCAL_PRESETS),
];
