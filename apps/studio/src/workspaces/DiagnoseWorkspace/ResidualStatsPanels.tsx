import { ErrorBox, Panel } from "@vitavision/ui";
import { api } from "../../api";
import type { RunRef } from "../../api/generated/types";
import { useQuery } from "../../session/query";
import { CameraResidualMatrix } from "./CameraResidualMatrix";
import { PoseStatsTable } from "./PoseStatsTable";

interface ResidualStatsPanelsProps {
  /** The run whose statistics (`residual_stats`) the panels show. */
  run: RunRef;
  selectedPose: number;
  selectedCamera: number;
  onSelectPose: (pose: number) => void;
  onSelectFrame: (pose: number, camera: number) => void;
}

/** The residual statistics side panels: per-pose residuals, and the
 * cross-camera matrix for a rig. Clicking a row or cell jumps there. */
export function ResidualStatsPanels({
  run,
  selectedPose,
  selectedCamera,
  onSelectPose,
  onSelectFrame,
}: ResidualStatsPanelsProps) {
  const stats = useQuery("residual_stats", { run }, () => api.residualStats({ run }));
  if (stats.error) return <ErrorBox>{stats.error.message}</ErrorBox>;
  if (!stats.data) return null;
  return (
    <>
      <Panel
        title="Per-pose residuals"
        actions={
          <span className="font-mono text-[10px] text-fg-muted">click a row to jump</span>
        }
      >
        <PoseStatsTable
          views={stats.data.views}
          selectedPose={selectedPose}
          onSelectPose={onSelectPose}
        />
      </Panel>
      {/* Renders nothing (not even its panel) for a single-camera export. */}
      <CameraResidualMatrix
        stats={stats.data}
        selectedPose={selectedPose}
        selectedCamera={selectedCamera}
        onSelect={onSelectFrame}
      />
    </>
  );
}
