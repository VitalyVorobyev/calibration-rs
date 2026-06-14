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
  /**
   * Optional config patch deep-merged over the topology's
   * `default_config_cmd` defaults when the preset is applied — for
   * datasets whose defaults are wrong (e.g. rtv3d needs Scheimpflug
   * sensors and EyeToHand). Keys mirror the Rust `*Config` JSON.
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

const RTV3D_MANIFEST_OVERRIDES = {
  detector: {
    chess_corners: {
      threshold_mode: "absolute",
      threshold_value: 30.0,
    },
  },
};

const RTV3D_JOINT_LASER_MANIFEST_OVERRIDES = {
  ...RTV3D_MANIFEST_OVERRIDES,
  topology: "rig_handeye_laserline",
  upstream_calibration: null,
};

const RTV3D_CAMERA_SEEDS = Array.from({ length: 6 }, () => ({
  fx: 2000.0,
  fy: 2000.0,
  cx: 360.0,
  cy: 270.0,
  skew: 0.0,
}));

const RTV3D_DISTORTION_SEEDS = Array.from({ length: 6 }, () => ({
  k1: 0.0,
  k2: 0.0,
  k3: 0.0,
  p1: 0.0,
  p2: 0.0,
  iters: 8,
}));

const RTV3D_SENSOR_SEEDS = Array.from({ length: 6 }, () => ({
  tilt_x: 0.0,
  tilt_y: 0.0,
}));

const RTV3D_HAND_EYE_CONFIG_OVERRIDES = {
  intrinsics: {
    fix_tangential: true,
    manual_init: {
      per_cam_intrinsics: RTV3D_CAMERA_SEEDS,
      per_cam_distortion: RTV3D_DISTORTION_SEEDS,
      per_cam_sensors: RTV3D_SENSOR_SEEDS,
    },
  },
  sensor: {
    kind: "Scheimpflug",
    init_tilt_x: 0.0,
    init_tilt_y: 0.0,
    fix_scheimpflug_in_intrinsics: { tilt_x: false, tilt_y: false },
    distortion_mask_in_percam_ba: {
      k1: false,
      k2: false,
      k3: true,
      p1: true,
      p2: true,
    },
    refine_scheimpflug_in_rig_ba: false,
  },
  rig: {
    refine_intrinsics_in_rig_ba: false,
  },
  handeye_init: { handeye_mode: "EyeToHand" },
  handeye_ba: { refine_robot_poses: true },
  solver: { max_iters: 200, robust_loss: { Huber: { scale: 1.0 } } },
};

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

  // ── Enabled: KUKA handeye (committed dataset) ────────────────────────────
  {
    id: "kuka-handeye",
    name: "KUKA handeye",
    group: "kuka_1 dataset",
    topology: "SingleCamHandeye",
    targetKind: "chessboard",
    targetSummary: "chessboard 17×28, 20 mm · robot poses 4×4",
    imageCount: 30,
    manifestPath: `${REPO_ROOT}/data/kuka_1/dataset.toml`,
  },

  // ── Enabled: rtv3d rig hand-eye (local-only dataset) ─────────────────────
  {
    id: "rtv3d-rig-handeye",
    name: "rtv3d rig hand-eye",
    group: "rtv3d 6-device Scheimpflug rig (local)",
    topology: "RigHandeye",
    targetKind: "charuco",
    targetSummary: "ChArUco 22×22, 5.2 mm · 6 tiled cameras · EyeToHand",
    imageCount: 20,
    manifestPath: `${REPO_ROOT}/privatedata/rtv3d/dataset_rig_handeye.toml`,
    manifestOverrides: RTV3D_MANIFEST_OVERRIDES,
    configOverrides: RTV3D_HAND_EYE_CONFIG_OVERRIDES,
  },

  // ── Enabled: rtv3d joint rig + laserline (V5 parity) ─────────────────────
  {
    id: "rtv3d-laser",
    name: "rtv3d laser planes",
    group: "rtv3d 6-device Scheimpflug rig (local)",
    topology: "RigHandeyeLaserline",
    targetKind: "charuco",
    targetSummary:
      "joint hand-eye + 6 laser planes · V5 path · EyeToHand",
    imageCount: 20,
    manifestPath: `${REPO_ROOT}/privatedata/rtv3d/dataset_laser.toml`,
    manifestOverrides: RTV3D_JOINT_LASER_MANIFEST_OVERRIDES,
    configOverrides: {
      handeye: RTV3D_HAND_EYE_CONFIG_OVERRIDES,
      laserline_init: {
        max_iters: 200,
        laser_residual_type: "PointToPlane",
      },
      joint_ba: {
        max_iters: 30,
        laser_residual_type: "PointToPlane",
        calib_weight: 1.0,
        laser_weight: 10000.0,
        refine_robot_poses: true,
        fix_first_camera_extrinsic: true,
        fix_scheimpflug_tilt: true,
        default_camera_fix: {
          intrinsics: { fx: false, fy: false, cx: true, cy: true },
          distortion: {
            k1: false,
            k2: false,
            k3: true,
            p1: true,
            p2: true,
          },
        },
      },
    },
  },

  // ── Enabled: rtv3d frozen rig laserline diagnostic (ADR 0021) ────────────
  {
    id: "rtv3d-laser-frozen",
    name: "rtv3d frozen laser diagnostic",
    group: "rtv3d 6-device Scheimpflug rig (local)",
    topology: "RigLaserlineDevice",
    targetKind: "charuco",
    targetSummary:
      "plane-only diagnostic over a frozen hand-eye export (requires rig_handeye_export.json)",
    imageCount: 20,
    manifestPath: `${REPO_ROOT}/privatedata/rtv3d/dataset_laser.toml`,
    manifestOverrides: RTV3D_MANIFEST_OVERRIDES,
    configOverrides: {
      max_iters: 200,
      laser_residual_type: "PointToPlane",
    },
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
    disabledReason:
      "Puzzleboard detection is now wired (B3C-PUZZLEBOARD); this card stays disabled until a DS8 puzzleboard manifest is committed — the DS8 data ships without a dataset.toml. Any puzzleboard/ringgrid dataset can already be run via the manifest editor.",
    milestone: "needs manifest",
  },
];
