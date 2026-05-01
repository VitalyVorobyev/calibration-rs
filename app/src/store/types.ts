import type { FrameKey, ImageManifest, PerFeatureResiduals } from "../types";

/** Discriminator for the loaded calibration export. Inferred from the
 * JSON shape until ts-rs codegen lands at B3.0 and pins this to a
 * Rust-side tag. */
export type ExportKind =
  | "planar_intrinsics"
  | "scheimpflug_intrinsics"
  | "single_cam_handeye"
  | "laserline_device"
  | "rig_extrinsics"
  | "rig_handeye"
  | "rig_laserline_device"
  | "unknown";

/** SE(3) wire format used across the workspace.
 *
 * `nalgebra::Isometry3` serializes as
 * `{ rotation: [qx, qy, qz, qw], translation: [tx, ty, tz] }`.
 * Quaternions are unit; the rotation list is `[i, j, k, w]` (Three.js's
 * convention). Translation is in meters. */
export interface Iso3Wire {
  rotation: [number, number, number, number];
  translation: [number, number, number];
}

/** Pinhole intrinsics block (`k` field of a serialized
 * `Camera<f64, Pinhole, BrownConrady5, IdentitySensor, FxFyCxCySkew>`). */
export interface FxFyCxCySkew {
  fx: number;
  fy: number;
  cx: number;
  cy: number;
  skew: number;
}

/** Brown-Conrady 5-parameter distortion. */
export interface BrownConrady5Wire {
  k1: number;
  k2: number;
  p1: number;
  p2: number;
  k3: number;
}

/** Subset of a pinhole `Camera<...>` JSON the 3D viewer reads. The full
 * struct also carries `proj` and `sensor`; both are `()` for pinhole +
 * identity sensor and irrelevant to the wireframe frustum. */
export interface PinholeCameraWire {
  k: FxFyCxCySkew;
  dist?: BrownConrady5Wire;
}

/** Loose union over the seven calibration export shapes. The viewer
 * (B1.0) only consumes the residuals + manifest + mean reprojection
 * error; rig-only fields (cameras, cam_se3_rig, rig_se3_target) are
 * optional and refined inside Viewer3DWorkspace / EpipolarWorkspace
 * once those phases land. */
export interface AnyExport {
  per_feature_residuals: PerFeatureResiduals;
  image_manifest?: ImageManifest;
  mean_reproj_error: number;
  /** Present on rig_extrinsics, rig_handeye, rig_laserline_device. */
  cameras?: PinholeCameraWire[];
  cam_se3_rig?: Iso3Wire[];
  rig_se3_target?: Iso3Wire[];
  /** Present on planar_intrinsics, scheimpflug_intrinsics, single_cam_handeye, laserline_device. */
  camera?: PinholeCameraWire;
  camera_se3_target?: Iso3Wire[];
  /** rig_handeye / single_cam_handeye carry mode-tagged hand-eye fields. */
  handeye_mode?: string;
  /** rig_laserline_device-only. */
  laser_planes_rig?: unknown[];
}

/** Tauri command response shape from `load_export`. */
export interface LoadExportResult {
  export: AnyExport;
  export_dir: string;
}

/** Re-export so callers can `import { FrameKey } from "../store/types"`. */
export type { FrameKey };
