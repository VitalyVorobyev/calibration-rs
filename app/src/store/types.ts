import type { FrameKey, ImageManifest, PerFeatureResiduals } from "../types";
import type {
  Camera,
  ExportKind as WireExportKind,
  Iso3Schema,
  LaserPlane,
} from "../types/generated/diagnose-wire";

// The export discriminator now lives next to `detectExportKind`, its single
// consumer-facing definition. Re-exported here so existing
// `import { ExportKind } from "./types"` sites keep resolving.
export type { ExportKind } from "./exportKind";

/** SE(3) wire format used across the workspace.
 *
 * Alias of the generated [`Iso3Schema`] — the single source of truth is the
 * Rust `Iso3` (`nalgebra::Isometry3`) serde shape:
 * `{ rotation: [qx, qy, qz, qw], translation: [tx, ty, tz] }`. Quaternions
 * are unit; the rotation list is `[i, j, k, w]` (Three.js's convention);
 * translation is in meters. */
export type Iso3Wire = Iso3Schema;

/** Serialized pinhole `Camera<f64, Pinhole, BrownConrady5, IdentitySensor,
 * FxFyCxCySkew>` — alias of the generated [`Camera`]. The 3D viewer only
 * reads `k` and `dist`; `proj` / `sensor` / `_phantom` are inert for the
 * pinhole + identity-sensor composition. */
export type PinholeCameraWire = Camera;

/** Laser plane wire format (`vision_calibration_optim::LaserPlane`) —
 * alias of the generated [`LaserPlane`]:
 * `{ normal: [nx, ny, nz], distance: d }` with a unit normal and the plane
 * equation `n · p + d = 0`. Frame depends on the carrying field
 * (`laser_planes_rig` = rig frame, `laser_planes_cam` = per-camera). */
export type LaserPlaneWire = LaserPlane;

/** Loose union over the seven calibration export shapes. The viewer
 * (B1.0) only consumes the residuals + manifest + mean reprojection
 * error; rig-only fields (cameras, cam_se3_rig, rig_se3_target) are
 * optional and refined inside Viewer3DWorkspace / EpipolarWorkspace
 * once those phases land. */
export interface AnyExport {
  /** Export-type discriminator. Present on every export the current
   * pipeline emits; typed optional only because `AnyExport` is a loose
   * structural view (a hand-edited payload or one written before the tag existed may omit it, in which
   * case `detectExportKind` returns `"unknown"`). */
  kind?: WireExportKind;
  per_feature_residuals: PerFeatureResiduals;
  image_manifest?: ImageManifest;
  /** All current exports carry this, but treat it as optional: exports
   * written before their type gained the field (e.g. older
   * rig_laserline_device) omit it on the wire. */
  mean_reproj_error?: number;
  /** Present on rig_extrinsics, rig_handeye, rig_handeye_laserline, rig_laserline_device. */
  cameras?: PinholeCameraWire[];
  cam_se3_rig?: Iso3Wire[];
  rig_se3_target?: Iso3Wire[];
  /** Present on planar_intrinsics, scheimpflug_intrinsics, single_cam_handeye, laserline_device. */
  camera?: PinholeCameraWire;
  camera_se3_target?: Iso3Wire[];
  /** rig_handeye / rig_handeye_laserline / single_cam_handeye carry mode-tagged hand-eye fields. */
  handeye_mode?: string;
  /** Laser rig exports: per-camera laser planes in rig frame. */
  laser_planes_rig?: LaserPlaneWire[];
  /** Laser rig exports: the same planes in each camera's frame. */
  laser_planes_cam?: LaserPlaneWire[];
}

/** Tauri command response shape from `load_export`. */
export interface LoadExportResult {
  export: AnyExport;
  export_dir: string;
}

/** Re-export so callers can `import { FrameKey } from "../store/types"`. */
export type { FrameKey };
