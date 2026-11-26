import { Panel } from "@vitavision/ui";
import type { TargetFeatureResidual } from "../../types";
import { CameraResidualMatrix } from "./CameraResidualMatrix";
import { PoseStatsTable } from "./PoseStatsTable";

interface ResidualStatsPanelsProps {
  residuals: TargetFeatureResidual[];
  selectedPose: number;
  selectedCamera: number;
  onSelectPose: (pose: number) => void;
  onSelectFrame: (pose: number, camera: number) => void;
}

/** The residual statistics side panels: per-pose residuals, and the
 * cross-camera matrix for a rig. Clicking a row or cell jumps there. */
export function ResidualStatsPanels({
  residuals,
  selectedPose,
  selectedCamera,
  onSelectPose,
  onSelectFrame,
}: ResidualStatsPanelsProps) {
  return (
    <>
      <Panel
        title="Per-pose residuals"
        actions={
          <span className="font-mono text-[10px] text-fg-muted">click a row to jump</span>
        }
      >
        <PoseStatsTable
          residuals={residuals}
          selectedPose={selectedPose}
          onSelectPose={onSelectPose}
        />
      </Panel>
      {/* Renders nothing (not even its panel) for a single-camera export. */}
      <CameraResidualMatrix
        residuals={residuals}
        selectedPose={selectedPose}
        selectedCamera={selectedCamera}
        onSelect={onSelectFrame}
      />
    </>
  );
}
