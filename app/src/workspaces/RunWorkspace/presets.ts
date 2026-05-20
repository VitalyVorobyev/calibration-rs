/** Built-in preset registry for the Run workspace (B3b).
 *
 * Each enabled preset carries the absolute path to its TOML manifest and
 * enough metadata to render a meaningful card without reading the file.
 * The manifest dir is derived automatically from the manifest path at
 * load time.
 *
 * DEV NOTE: Manifest paths are hard-coded against the repo root at
 * `/Users/vitalyvorobyev/vision/calibration-rs`. This is intentional
 * scaffolding for local development; B3d will replace it with
 * bundle-asset resolution via Tauri's `resource_dir` API.
 */

// Absolute path to the repo root — replace this constant in B3d when
// bundle-asset resolution ships.
const REPO_ROOT = "/Users/vitalyvorobyev/vision/calibration-rs";

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
  /** Absolute path to the TOML manifest file. */
  manifestPath: string;
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
  /** Release milestone badge text. */
  milestone: string;
}

export type Preset = EnabledPreset | DisabledPreset;

export const BUILTIN_PRESETS: Preset[] = [
  // ── Enabled: stereo-left ────────────────────────────────────────────────
  {
    id: "stereo-left",
    name: "Stereo · cam-left",
    group: "Bundled stereo dataset",
    topology: "PlanarIntrinsics",
    targetKind: "chessboard",
    targetSummary: "chessboard 7×11, 30 mm",
    imageCount: 20,
    manifestPath: `${REPO_ROOT}/data/stereo/dataset_left.toml`,
  },

  // ── Enabled: stereo-right ───────────────────────────────────────────────
  {
    id: "stereo-right",
    name: "Stereo · cam-right",
    group: "Bundled stereo dataset",
    topology: "PlanarIntrinsics",
    targetKind: "chessboard",
    targetSummary: "chessboard 7×11, 30 mm",
    imageCount: 20,
    manifestPath: `${REPO_ROOT}/data/stereo/dataset_right.toml`,
  },

  // ── Disabled: stereo rig ────────────────────────────────────────────────
  {
    id: "stereo-rig",
    disabled: true,
    name: "Stereo rig",
    group: "Bundled stereo dataset",
    topology: "RigExtrinsics",
    targetKind: "chessboard",
    targetSummary: "chessboard 7×11, 30 mm",
    imageCount: null,
    disabledReason: "Multi-camera rig topology ships in B3c.",
    milestone: "B3c",
  },

  // ── Disabled: stereo-charuco ────────────────────────────────────────────
  {
    id: "stereo-charuco",
    disabled: true,
    name: "Stereo · ChArUco",
    group: "Bundled stereo dataset",
    topology: "PlanarIntrinsics",
    targetKind: "charuco",
    targetSummary: "ChArUco DICT_4X4_50",
    imageCount: null,
    disabledReason: "ChArUco detector integration ships in B3c.",
    milestone: "B3c",
  },

  // ── Disabled: KUKA handeye ───────────────────────────────────────────────
  {
    id: "kuka-handeye",
    disabled: true,
    name: "KUKA handeye",
    group: "kuka_1 dataset",
    topology: "SingleCamHandeye",
    targetKind: "chessboard",
    targetSummary: "chessboard, robot poses CSV",
    imageCount: null,
    disabledReason: "Hand-eye calibration topology ships in B3c.",
    milestone: "B3c",
  },

  // ── Disabled: DS8 Scheimpflug ────────────────────────────────────────────
  {
    id: "ds8-scheimpflug",
    disabled: true,
    name: "DS8 Scheimpflug",
    group: "DS8 dataset",
    topology: "ScheimpflugIntrinsics",
    targetKind: "puzzleboard",
    targetSummary: "puzzleboard puzzle_130x130",
    imageCount: null,
    disabledReason: "Scheimpflug topology + puzzleboard detector ship in B3c.",
    milestone: "B3c",
  },
];
