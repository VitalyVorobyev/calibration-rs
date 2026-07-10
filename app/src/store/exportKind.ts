import type {
  LaserlineDeviceExport,
  PlanarIntrinsicsExport,
  RigExtrinsicsExport,
  RigHandeyeExport,
  RigHandeyeLaserlineExport,
  RigLaserlineDeviceExport,
  ScheimpflugIntrinsicsExport,
  SingleCamHandeyeExport,
} from "../types/generated/diagnose-wire";

/** Discriminated union over every calibration export the diagnose app can
 * load. Members are the generated (Rust-sourced) `*Export` interfaces, so
 * this stays in lockstep with the pipeline via `bun run generate:types`. */
export type CalibrationExport =
  | PlanarIntrinsicsExport
  | ScheimpflugIntrinsicsExport
  | SingleCamHandeyeExport
  | LaserlineDeviceExport
  | RigExtrinsicsExport
  | RigHandeyeExport
  | RigLaserlineDeviceExport
  | RigHandeyeLaserlineExport;

/** Discriminator for the loaded calibration export. */
export type ExportKind =
  | "planar_intrinsics"
  | "scheimpflug_intrinsics"
  | "single_cam_handeye"
  | "laserline_device"
  | "rig_extrinsics"
  | "rig_handeye"
  | "rig_handeye_laserline"
  | "rig_laserline_device"
  | "unknown";

// Distinguishing field names, pinned to the generated interfaces via
// `satisfies keyof …`. If a Rust field is renamed and the types are
// regenerated, these stop compiling — which is the whole point: the
// discriminator can no longer silently drift from the export shape.
const F_HANDEYE_MODE = "handeye_mode" satisfies keyof RigHandeyeExport &
  keyof SingleCamHandeyeExport &
  keyof RigHandeyeLaserlineExport;
const F_CAMERAS = "cameras" satisfies keyof RigExtrinsicsExport & keyof RigHandeyeExport;
const F_CAMERA = "camera" satisfies keyof SingleCamHandeyeExport;
const F_LASER_PLANES_RIG = "laser_planes_rig" satisfies keyof RigLaserlineDeviceExport &
  keyof RigHandeyeLaserlineExport;
const F_ESTIMATE = "estimate" satisfies keyof LaserlineDeviceExport;
const F_STATS = "stats" satisfies keyof LaserlineDeviceExport;
const F_PARAMS = "params" satisfies keyof PlanarIntrinsicsExport &
  keyof ScheimpflugIntrinsicsExport;

/** Narrows `value` to a plain (non-null, non-array) object — guards the
 * `params` cast below against malformed exports where the field exists
 * but isn't a record (e.g. a string or array). */
function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Sensor-model tag inside `*IntrinsicsExport.params.camera.sensor`, the only
 * place planar and Scheimpflug single-camera exports differ (their top-level
 * shapes are identical). */
function intrinsicsSensorTag(params: Record<string, unknown>): string | undefined {
  const camera = params.camera as { sensor?: { type?: unknown } } | undefined;
  const tag = camera?.sensor?.type;
  return typeof tag === "string" ? tag : undefined;
}

/** Classify a loaded export by its distinguishing required fields.
 *
 * Grounded in the generated `*Export` shapes rather than the JSON-sniffing
 * of the retired `exportShape.ts`: those probes read top-level `camera` /
 * `laser_planes_cam` fields that the real exports never carry (they live
 * under `params` / `estimate`). Probe order runs most-specific first —
 * laser-rig before rig, rig before single-camera. */
export function detectExportKind(data: unknown): ExportKind {
  if (data == null || typeof data !== "object") return "unknown";
  const d = data as Record<string, unknown>;
  const has = (k: string) => d[k] != null;

  // Rig + laser flavours carry per-camera laser planes in the rig frame.
  if (has(F_LASER_PLANES_RIG)) {
    return has(F_HANDEYE_MODE) ? "rig_handeye_laserline" : "rig_laserline_device";
  }
  // Single-camera laser device: bundle estimate + laser stats, no rig planes.
  if (has(F_ESTIMATE) && has(F_STATS)) return "laserline_device";
  // Multi-camera rig: a `cameras` array (with or without hand-eye).
  if (has(F_CAMERAS)) {
    return has(F_HANDEYE_MODE) ? "rig_handeye" : "rig_extrinsics";
  }
  // Single-camera hand-eye: one `camera` plus a hand-eye mode.
  if (has(F_CAMERA) && has(F_HANDEYE_MODE)) return "single_cam_handeye";
  // Single-camera intrinsics: `params`; planar vs Scheimpflug by sensor model.
  // `params` must be a plain object — a malformed export where it's some
  // other JSON type (string, array, …) falls through to "unknown" instead
  // of silently defaulting to planar.
  const params = d[F_PARAMS];
  if (isPlainObject(params)) {
    return intrinsicsSensorTag(params) === "scheimpflug"
      ? "scheimpflug_intrinsics"
      : "planar_intrinsics";
  }
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
    case "rig_handeye_laserline":
      return "Rig hand-eye + laserline";
    case "rig_laserline_device":
      return "Rig + laserline";
    case "unknown":
      return "Unknown export";
  }
}
