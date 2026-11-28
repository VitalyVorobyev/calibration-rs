import { Table, type Column } from "@vitavision/ui";
import { useMemo } from "react";
import type {
  PlaneDifference,
  PoseDifference,
  RunComparison,
} from "../../api/generated/types";
import { formatLength, significant } from "../../lib/format";
import { positionName, signed } from "./compareModel";
import { TestCell } from "./TestCell";

interface PoseRow {
  label: string;
  /** One per run after the reference. */
  others: (PoseDifference | null)[];
}

interface PlaneRow {
  camera: string;
  others: (PlaneDifference | null)[];
}

/** The blocks that are not a camera's parameters: each camera's pose
 * against the reference run's rig camera, the hand-eye, and the laser
 * planes. Per-view poses and the target's pose describe a capture, not the
 * device, and are never compared. */
export function PoseBlocks({ comparison }: { comparison: RunComparison }) {
  const names = comparison.runs.map((r) => r.name);
  const diffs = comparison.differences;
  const rig = useMemo(() => poseRows(diffs.map((d) => d.rig)), [diffs]);
  const handeye = useMemo(
    () => poseRows(diffs.map((d) => (d.handeye ? [d.handeye] : []))),
    [diffs],
  );
  const planes = useMemo(() => {
    const cams: string[] = [];
    for (const d of diffs)
      for (const p of d.laser_planes) if (!cams.includes(p.camera)) cams.push(p.camera);
    return cams.map((camera): PlaneRow => ({
      camera,
      others: diffs.map((d) => d.laser_planes.find((p) => p.camera === camera) ?? null),
    }));
  }, [diffs]);
  const poseColumns = useMemo(() => buildPoseColumns(names), [names]);
  const planeColumns = useMemo(() => buildPlaneColumns(names), [names]);

  return (
    <>
      {rig.length > 0 && (
        <Block
          title="Rig poses"
          note="Each camera's pose relative to the reference run's rig camera, compared whichever camera each run held fixed."
        >
          <Table
            caption="Rig poses"
            columns={poseColumns}
            rows={rig}
            rowKey={(r) => r.label}
            pinFirstColumn
          />
        </Block>
      )}
      {handeye.length > 0 && (
        <Block
          title="Hand-eye"
          note="The hand-eye, expressed for the reference run's rig camera."
        >
          <Table
            caption="Hand-eye"
            columns={poseColumns}
            rows={handeye}
            rowKey={(r) => r.label}
            pinFirstColumn
          />
        </Block>
      )}
      {planes.length > 0 && (
        <Block
          title="Laser planes"
          note="The angle between the planes' normals and the change of the plane's distance, per camera."
        >
          <Table
            caption="Laser planes"
            columns={planeColumns}
            rows={planes}
            rowKey={(r) => r.camera}
            pinFirstColumn
          />
        </Block>
      )}
    </>
  );
}

function Block({
  title,
  note,
  children,
}: {
  title: string;
  note: string;
  children: React.ReactNode;
}) {
  return (
    <section aria-label={title} className="flex flex-col gap-1.5">
      <h3 className="text-sm text-fg">{title}</h3>
      <p className="text-xs text-fg-muted">{note}</p>
      {children}
    </section>
  );
}

function poseRows(perRun: PoseDifference[][]): PoseRow[] {
  const labels: string[] = [];
  for (const run of perRun)
    for (const p of run) if (!labels.includes(p.label)) labels.push(p.label);
  return labels.map((label) => ({
    label,
    others: perRun.map((run) => run.find((p) => p.label === label) ?? null),
  }));
}

const ABSENT = "not in this run";

function buildPoseColumns(names: string[]): Column<PoseRow>[] {
  const cols: Column<PoseRow>[] = [
    {
      key: "label",
      header: "Pose",
      cell: (r) => <span className="font-mono text-xs">{r.label}</span>,
    },
  ];
  names.slice(1).forEach((name, k) => {
    const group = `${positionName(k + 1)} · ${name}`;
    cols.push(
      {
        key: `r${k}`,
        header: "Rotation",
        group,
        numeric: true,
        sortable: true,
        firstSort: "descending",
        sortValue: (r) => r.others[k]?.rotation_deg ?? null,
        cell: (r) =>
          r.others[k] ? `${significant(r.others[k].rotation_deg, 3)}°` : ABSENT,
      },
      {
        key: `t${k}`,
        header: "Translation",
        group,
        numeric: true,
        sortable: true,
        firstSort: "descending",
        sortValue: (r) => r.others[k]?.translation_norm_m ?? null,
        cell: (r) =>
          r.others[k] ? formatLength(r.others[k].translation_norm_m) : ABSENT,
      },
      {
        key: `s${k}`,
        header: "Δ in σ",
        group,
        numeric: true,
        cell: (r) => {
          const v = r.others[k]?.delta_over_reference_sigma;
          return v == null ? "–" : signed(v);
        },
      },
      {
        key: `x${k}`,
        header: "6-dof test",
        group,
        cell: (r) => (r.others[k] ? <TestCell test={r.others[k].test} /> : null),
      },
    );
  });
  return cols;
}

function buildPlaneColumns(names: string[]): Column<PlaneRow>[] {
  const cols: Column<PlaneRow>[] = [
    {
      key: "camera",
      header: "Camera",
      cell: (r) => <span className="font-mono text-xs">{r.camera}</span>,
    },
  ];
  names.slice(1).forEach((name, k) => {
    const group = `${positionName(k + 1)} · ${name}`;
    cols.push(
      {
        key: `a${k}`,
        header: "Normal angle",
        group,
        numeric: true,
        cell: (r) =>
          r.others[k] ? `${significant(r.others[k].normal_angle_deg, 3)}°` : ABSENT,
      },
      {
        key: `d${k}`,
        header: "Distance Δ",
        group,
        numeric: true,
        cell: (r) => (r.others[k] ? formatLength(r.others[k].distance_delta_m) : ABSENT),
      },
      {
        key: `t${k}`,
        header: "Distance test",
        group,
        cell: (r) => (r.others[k] ? <TestCell test={r.others[k].distance_test} /> : null),
      },
    );
  });
  return cols;
}
