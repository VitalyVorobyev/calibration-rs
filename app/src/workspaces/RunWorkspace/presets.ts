/** Built-in preset registry for the Run workspace.
 *
 * Each enabled preset carries the absolute path to its TOML manifest and
 * enough metadata to render a meaningful card without reading the file.
 * The manifest dir is derived automatically from the manifest path at
 * load time.
 *
 * DEV NOTE: Manifest paths are hard-coded against the repo root at
 * `/Users/vitalyvorobyev/vision/calibration-rs`. This is intentional
 * scaffolding for local development, to be replaced with bundle-asset
 * resolution via Tauri's `resource_dir` API.
 */

// Absolute path to the repo root — replace this constant when
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
  /** Short badge text for the planned capability. */
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

  // ── Enabled: stereo rig ─────────────────────────────────────────────────
  {
    id: "stereo-rig",
    name: "Stereo rig",
    group: "Bundled stereo dataset",
    topology: "RigExtrinsics",
    targetKind: "chessboard",
    targetSummary: "chessboard 7×11, 30 mm · 2 cameras",
    imageCount: 20,
    manifestPath: `${REPO_ROOT}/data/stereo/dataset_rig.toml`,
  },

  // ── Enabled: stereo-charuco (single camera) ─────────────────────────────
  {
    id: "stereo-charuco",
    name: "ChArUco · cam1",
    group: "Stereo ChArUco dataset",
    topology: "PlanarIntrinsics",
    targetKind: "charuco",
    targetSummary: "ChArUco 22×22, 1.35 mm, DICT_4X4_1000",
    imageCount: 28,
    manifestPath: `${REPO_ROOT}/data/stereo_charuco/dataset_cam1.toml`,
  },

  // ── Enabled: stereo-charuco rig (token pairing) ─────────────────────────
  {
    id: "stereo-charuco-rig",
    name: "ChArUco rig",
    group: "Stereo ChArUco dataset",
    topology: "RigExtrinsics",
    targetKind: "charuco",
    targetSummary: "ChArUco 22×22 · 2 cameras · token pairing",
    imageCount: 27,
    manifestPath: `${REPO_ROOT}/data/stereo_charuco/dataset_rig.toml`,
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
    disabledReason: "Hand-eye calibration topology is not wired into Run yet.",
    milestone: "planned",
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
    disabledReason: "The puzzleboard detector is not wired into Run yet (the Scheimpflug topology itself is).",
    milestone: "planned",
  },
];
