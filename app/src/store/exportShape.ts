import type { AnyExport, ExportKind } from "./types";

/** Infer the calibration export kind from JSON shape.
 *
 * Bridge solution until ts-rs codegen lands at B3.0 — at that point
 * each `*Export` will carry a discriminator tag and this function is
 * replaced by a one-line lookup. The probe order matters: rig-shaped
 * exports must be checked before single-cam shapes because some rig
 * exports also expose top-level `camera`-prefixed legacy fields. */
export function inferExportKind(data: AnyExport): ExportKind {
  const d = data as unknown as Record<string, unknown>;
  if (Array.isArray(d.laser_planes_rig)) return "rig_laserline_device";
  if (Array.isArray(d.cameras) && d.handeye_mode != null) return "rig_handeye";
  if (Array.isArray(d.cameras)) return "rig_extrinsics";
  if (d.handeye_mode != null) return "single_cam_handeye";
  if (Array.isArray(d.laser_planes_cam) || d.laser_plane_cam != null) {
    return "laserline_device";
  }
  // Scheimpflug single-cam intrinsics carries `sensor` (Scheimpflug tilt)
  // alongside `camera`; planar does not.
  if (d.camera != null && (d.sensor != null || d.scheimpflug != null)) {
    return "scheimpflug_intrinsics";
  }
  if (d.camera != null) return "planar_intrinsics";
  return "unknown";
}

/** Human-readable label for an export kind, for the diagnose header. */
export function exportKindLabel(kind: ExportKind): string {
  switch (kind) {
    case "planar_intrinsics":
      return "Planar intrinsics";
    case "scheimpflug_intrinsics":
      return "Scheimpflug intrinsics";
    case "single_cam_handeye":
      return "Single-cam hand-eye";
    case "laserline_device":
      return "Laserline device";
    case "rig_extrinsics":
      return "Rig extrinsics";
    case "rig_handeye":
      return "Rig hand-eye";
    case "rig_laserline_device":
      return "Rig + laserline";
    case "unknown":
      return "Unknown export";
  }
}
