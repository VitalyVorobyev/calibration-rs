import { useState } from "react";
import type { RunRef } from "../../api/generated/types";
import { useSelection } from "../../session/selection";
import { CameraFrame } from "./CameraFrame";
import { ResidualMapPanel } from "./maps/ResidualMapPanel";
import { useResidualMap } from "./maps/useMaps";
import { ResidualRecords } from "./ResidualRecords";
import { ResidualStatsTable } from "./ResidualStatsTable";
import { scopeViews, type ViewScope } from "./residualModel";

const NO_OUTLIERS: number[] = [];

/** Where the model and the data disagree: the residual map of the camera
 * over the chosen views, below it every feature's residual as a scatter and
 * a table, and the run's reprojection statistics. */
export function ResidualsTab() {
  return (
    <CameraFrame title="Residuals">
      {({ run, camera }) => <Body key={camera} run={run} camera={camera} />}
    </CameraFrame>
  );
}

function Body({ run, camera }: { run: RunRef; camera: number }) {
  const [scope, setScope] = useState<ViewScope>({ kind: "all" });
  const selectedView = useSelection((s) => s.selection.view);
  const views = scopeViews(scope, selectedView);
  const { map, state } = useResidualMap(run, camera, views);
  return (
    <div className="flex min-h-0 flex-1 flex-col gap-6 overflow-y-auto pr-1">
      <div className="flex min-h-[32rem] shrink-0 flex-col">
        <ResidualMapPanel
          run={run}
          camera={camera}
          scope={scope}
          onScope={setScope}
          map={map}
          state={state}
        />
      </div>
      <div className="shrink-0">
        <ResidualRecords
          run={run}
          camera={camera}
          views={views}
          outliers={map?.outliers ?? NO_OUTLIERS}
        />
      </div>
      <div className="shrink-0">
        <ResidualStatsTable run={run} />
      </div>
    </div>
  );
}
