/** Topology registry for the Run workspace.
 *
 * Maps every `DatasetSpec.topology` wire name to its display label,
 * schemars-emitted config schema, and whether the in-process runner
 * supports it today. The Run button is disabled (with a reason) for
 * unsupported topologies, mirroring the dispatch in
 * `app/src-tauri/src/run.rs`.
 */
import type { JsonSchema } from "../../lib/configForm";

import planarConfigSchemaJson from "../../schemas/planar_intrinsics_config.json";
import rigExtrinsicsConfigSchemaJson from "../../schemas/rig_extrinsics_config.json";
import rigHandeyeConfigSchemaJson from "../../schemas/rig_handeye_config.json";
import scheimpflugConfigSchemaJson from "../../schemas/scheimpflug_intrinsics_config.json";

// schemars-emitted JSON Schemas; cast through unknown since both shapes
// are JSON-compatible (our JsonSchema interface is intentionally loose).
const planarConfigSchema = planarConfigSchemaJson as unknown as JsonSchema;
const scheimpflugConfigSchema = scheimpflugConfigSchemaJson as unknown as JsonSchema;
const rigExtrinsicsConfigSchema = rigExtrinsicsConfigSchemaJson as unknown as JsonSchema;
const rigHandeyeConfigSchema = rigHandeyeConfigSchemaJson as unknown as JsonSchema;

export interface TopologyInfo {
  /** Human-readable label for headers and summaries. */
  label: string;
  /** Config schema driving the ConfigForm; null when unsupported. */
  schema: JsonSchema | null;
  /** Whether the Tauri runner can execute this topology today. */
  supported: boolean;
  /** Shown as the Run-button tooltip when unsupported. */
  unsupportedReason?: string;
}

/** All seven `Topology` wire names (serde snake_case). */
export const TOPOLOGY_INFO: Record<string, TopologyInfo> = {
  planar_intrinsics: {
    label: "PlanarIntrinsics",
    schema: planarConfigSchema,
    supported: true,
  },
  scheimpflug_intrinsics: {
    label: "ScheimpflugIntrinsics",
    schema: scheimpflugConfigSchema,
    supported: true,
  },
  single_cam_handeye: {
    label: "SingleCamHandeye",
    schema: null,
    supported: false,
    unsupportedReason: "SingleCamHandeye lands in B3c-2.",
  },
  laserline_device: {
    label: "LaserlineDevice",
    schema: null,
    supported: false,
    unsupportedReason: "Laser topologies await the laser-frame manifest design.",
  },
  rig_extrinsics: {
    label: "RigExtrinsics",
    schema: rigExtrinsicsConfigSchema,
    supported: true,
  },
  rig_handeye: {
    label: "RigHandeye",
    schema: rigHandeyeConfigSchema,
    supported: true,
  },
  rig_laserline_device: {
    label: "RigLaserlineDevice",
    schema: null,
    supported: false,
    unsupportedReason: "Laser topologies await the laser-frame manifest design.",
  },
};

/** Lookup with a graceful fallback for unknown / misspelled topologies. */
export function topologyInfo(topology: string): TopologyInfo {
  return (
    TOPOLOGY_INFO[topology] ?? {
      label: topology || "unknown",
      schema: null,
      supported: false,
      unsupportedReason: `Unknown topology "${topology}" — check the manifest.`,
    }
  );
}
