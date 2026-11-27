import type {
  ProblemKind as WireExportKind,
  LaserlineDeviceExport,
  PlanarIntrinsicsExport,
  RigExtrinsicsExport,
  RigHandeyeExport,
  RigHandeyeLaserlineExport,
  RigLaserlineDeviceExport,
  SingleCamHandeyeExport,
} from "../api/generated/types";

/** Discriminated union over every calibration export the app can
 * load. Members are the generated (Rust-sourced) `*Export` interfaces, each
 * carrying its own `kind` discriminator, so this stays in lockstep with
 * the pipeline via `bun run generate:types`. */
export type CalibrationExport =
  | PlanarIntrinsicsExport
  | SingleCamHandeyeExport
  | LaserlineDeviceExport
  | RigExtrinsicsExport
  | RigHandeyeExport
  | RigLaserlineDeviceExport
  | RigHandeyeLaserlineExport;

/** Discriminator for the loaded calibration export: the seven wire kinds
 * (from the generated {@link WireExportKind}) plus a UI-only `"unknown"`
 * sentinel for payloads that carry no recognised tag. */
export type ExportKind = WireExportKind | "unknown";

/** The one source of a workflow's name as people say it (badges, filters,
 * headers, the editor's subtitle), keyed by wire kind.
 *
 * This `Record` is the single source of truth for both the label vocabulary
 * *and* the set of recognised kinds (see {@link detectExportKind}). Typing it
 * `Record<WireExportKind, string>` pins it to the generated union: adding or
 * renaming a Rust `ProblemKind` variant and regenerating the types forces a
 * matching edit here, or the build fails. */
const WIRE_KIND_LABELS: Record<WireExportKind, string> = {
  planar_intrinsics: "Planar intrinsics",
  single_cam_handeye: "Single-camera hand-eye",
  laserline_device: "Laser-line device",
  rig_extrinsics: "Rig extrinsics",
  rig_handeye: "Rig hand-eye",
  rig_laserline_device: "Rig laser-line device",
  rig_handeye_laserline: "Rig hand-eye with laser line",
};

/** Narrows an arbitrary string to a known wire kind. */
function isWireExportKind(kind: string): kind is WireExportKind {
  return Object.prototype.hasOwnProperty.call(WIRE_KIND_LABELS, kind);
}

/** Classify a loaded export by its `kind` discriminator.
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
