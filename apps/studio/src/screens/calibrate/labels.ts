import type { ProblemKind } from "../../api/generated/types";

const TOPOLOGY: Record<ProblemKind, string> = {
  planar_intrinsics: "Planar intrinsics",
  single_cam_handeye: "Single-camera hand-eye",
  laserline_device: "Laser-line device",
  rig_extrinsics: "Rig extrinsics",
  rig_handeye: "Rig hand-eye",
  rig_handeye_laserline: "Rig hand-eye with laser line",
  rig_laserline_device: "Rig laser-line device",
};

/** A workflow's name as people say it. */
export function topologyLabel(kind: ProblemKind): string {
  return TOPOLOGY[kind] ?? kind.replace(/_/g, " ");
}
