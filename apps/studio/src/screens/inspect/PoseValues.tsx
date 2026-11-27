import { Table, type Column } from "@vitavision/ui";
import type { PlaneParameter, PoseParameter } from "../../api/generated/types";
import { planeView, poseView, type PlaneView, type PoseView } from "./parameterModel";

const mono = (text: string) => <span className="font-mono text-fg">{text}</span>;

const POSE_COLUMNS: Column<PoseView>[] = [
  { key: "label", header: "Pose", cell: (r) => mono(r.label) },
  {
    key: "translation",
    header: "Translation (mm)",
    numeric: true,
    cell: (r) => <span data-testid={`pose-${r.label}`}>{r.translation}</span>,
  },
  { key: "rotation", header: "Rotation (°)", numeric: true, cell: (r) => r.rotation },
  {
    key: "sigma-r",
    header: "σ about (°)",
    numeric: true,
    cell: (r) => <span title={r.reason ?? undefined}>{r.sigmaRotation}</span>,
  },
  {
    key: "sigma-t",
    header: "σ along (mm)",
    numeric: true,
    cell: (r) => <span title={r.reason ?? undefined}>{r.sigmaTranslation}</span>,
  },
  {
    key: "note",
    header: "Note",
    cell: (r) => <span className="text-xs text-fg-muted">{r.reason ?? ""}</span>,
  },
];

const PLANE_COLUMNS: Column<PlaneView>[] = [
  { key: "label", header: "Plane", cell: (r) => mono(r.label) },
  { key: "normal", header: "Normal ± σ", numeric: true, cell: (r) => r.normal },
  {
    key: "distance",
    header: "Distance ± σ (mm)",
    numeric: true,
    cell: (r) => r.distance,
  },
  {
    key: "note",
    header: "Note",
    cell: (r) => <span className="text-xs text-fg-muted">{r.reason ?? ""}</span>,
  },
];

/** The poses the run estimates for the camera (its pose in the rig) and for
 * the rig (the hand-eye): value and the σ of the perturbation the
 * covariance is over. */
export function PoseValues({ poses }: { poses: readonly PoseParameter[] }) {
  if (poses.length === 0) return null;
  return (
    <section aria-label="Poses" className="flex flex-col gap-1">
      <h3 className="text-[11px] uppercase tracking-wider text-fg-muted">Poses</h3>
      <Table
        caption="Pose values"
        columns={POSE_COLUMNS}
        rows={poses.map(poseView)}
        rowKey={(r) => r.label}
      />
      <p className="text-xs text-fg-muted">
        A pose <span className="font-mono">to_se3_from</span> is the translation of the
        from frame&apos;s origin in the to frame and an axis-angle rotation. Its σ is of a
        small rotation about and translation along the from frame&apos;s axes, not of
        these components.
      </p>
    </section>
  );
}

/** The camera's laser plane in its own frame: normal and distance, each
 * with its σ. */
export function PlaneValues({ planes }: { planes: readonly PlaneParameter[] }) {
  if (planes.length === 0) return null;
  return (
    <section aria-label="Laser plane" className="flex flex-col gap-1">
      <h3 className="text-[11px] uppercase tracking-wider text-fg-muted">Laser plane</h3>
      <Table
        caption="Laser plane values"
        columns={PLANE_COLUMNS}
        rows={planes.map(planeView)}
        rowKey={(r) => r.label}
      />
    </section>
  );
}
