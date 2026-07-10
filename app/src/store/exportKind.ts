import type {
  ExportKind as WireExportKind,
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
 * load. Members are the generated (Rust-sourced) `*Export` interfaces, each
 * carrying its own `kind` discriminator (R7), so this stays in lockstep with
 * the pipeline via `bun run generate:types`. */
export type CalibrationExport =
  | PlanarIntrinsicsExport
  | ScheimpflugIntrinsicsExport
  | SingleCamHandeyeExport
  | LaserlineDeviceExport
  | RigExtrinsicsExport
  | RigHandeyeExport
  | RigLaserlineDeviceExport
  | RigHandeyeLaserlineExport;

/** Discriminator for the loaded calibration export: the eight wire kinds
 * (from the generated {@link WireExportKind}) plus a UI-only `"unknown"`
 * sentinel for payloads that carry no recognised tag. */
export type ExportKind = WireExportKind | "unknown";

/** Human-readable labels for the diagnose header, keyed by wire kind.
 *
 * This `Record` is the single source of truth for both the label vocabulary
 * *and* the set of recognised kinds (see {@link detectExportKind}). Typing it
 * `Record<WireExportKind, string>` pins it to the generated union: adding or
 * renaming a Rust `ExportKind` variant and regenerating the types forces a
 * matching edit here, or the build fails. */
const WIRE_KIND_LABELS: Record<WireExportKind, string> = {
  planar_intrinsics: "Planar intrinsics",
  scheimpflug_intrinsics: "Scheimpflug intrinsics",
  single_cam_handeye: "Single-cam hand-eye",
  laserline_device: "Laserline device",
  rig_extrinsics: "Rig extrinsics",
  rig_handeye: "Rig hand-eye",
  rig_laserline_device: "Rig + laserline",
  rig_handeye_laserline: "Rig hand-eye + laserline",
};

/** Narrows an arbitrary string to a known wire kind. */
function isWireExportKind(kind: string): kind is WireExportKind {
  return Object.prototype.hasOwnProperty.call(WIRE_KIND_LABELS, kind);
}

/** Classify a loaded export by its `kind` discriminator (R7).
 *
 * Every pipeline `*Export` serializes a required `kind` tag, so classification
 * is a single field read validated against the known vocabulary — no more
 * probing which required fields happen to be present. A payload that is not an
 * object, or whose `kind` is missing / unrecognised, is `"unknown"`. */
export function detectExportKind(data: unknown): ExportKind {
  if (data == null || typeof data !== "object") return "unknown";
  const kind = (data as { kind?: unknown }).kind;
  return typeof kind === "string" && isWireExportKind(kind) ? kind : "unknown";
}

/** Human-readable label for an export kind, for the diagnose header. */
export function exportKindLabel(kind: ExportKind): string {
  return kind === "unknown" ? "Unknown export" : WIRE_KIND_LABELS[kind];
}
