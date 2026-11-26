/** Topology registry for the Run workspace: the display label of every
 * `DatasetSpec.topology` wire name. The forms' schemas come from the back
 * end (`config_form`), so an unknown topology is one it has no form for. */

export interface TopologyInfo {
  /** Human-readable label for headers and summaries. */
  label: string;
  /** Whether the runner can execute this topology. */
  supported: boolean;
  /** Shown as the Run-button tooltip when unsupported. */
  unsupportedReason?: string;
}

/** All seven `Topology` wire names (serde snake_case) and their labels. */
const LABELS: Record<string, string> = {
  planar_intrinsics: "PlanarIntrinsics",
  single_cam_handeye: "SingleCamHandeye",
  laserline_device: "LaserlineDevice",
  rig_extrinsics: "RigExtrinsics",
  rig_handeye: "RigHandeye",
  rig_handeye_laserline: "RigHandeyeLaserline",
  rig_laserline_device: "RigLaserlineDevice",
};

/** Lookup with a graceful fallback for unknown / misspelled topologies. */
export function topologyInfo(topology: string): TopologyInfo {
  const label = LABELS[topology];
  if (label !== undefined) return { label, supported: true };
  return {
    label: topology || "unknown",
    supported: false,
    unsupportedReason: `Unknown topology "${topology}" — check the manifest.`,
  };
}
