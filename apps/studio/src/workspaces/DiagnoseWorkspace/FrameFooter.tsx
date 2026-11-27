import { useMemo } from "react";
import { laserRamp, targetRamp } from "../../lib/errorColors";
import { useStore } from "../../store";
import type { AnyExport } from "../../store/types";
import type { CursorReadout, FrameKey, LaserFeatureResidual } from "../../types";
import { CursorChip, LaserResidualLegend, ResidualLegend } from "./ResidualLegends";
import { RoiHistogram } from "./RoiHistogram";

interface FrameFooterProps {
  data: AnyExport;
  frame: FrameKey;
  /** The laser frame and its residuals are shown, not the target's. */
  laser: boolean;
  /** Every laser residual of the run (the frame's are picked here). */
  laserResiduals: LaserFeatureResidual[];
  camera: number;
  /** The image's luminance histogram; `null` while it decodes. */
  histogram: number[] | null;
  cursor: CursorReadout | null;
}

/** Under the frame: its residual legend and, when the frame has an image,
 * the luminance histogram and the cursor readout. */
export function FrameFooter({
  data,
  frame,
  laser,
  laserResiduals: laserAll,
  camera,
  histogram,
  cursor,
}: FrameFooterProps) {
  const run = useStore((s) => s.run);
  const inFrame = (r: { view: number; camera: number }) =>
    r.view === frame.pose && r.camera === frame.camera;
  const residuals = data.diagnostics.per_feature_residuals;
  // The same scales the canvas draws on, from the same run-wide arrays.
  const targetScale = useMemo(() => targetRamp(residuals.target), [residuals.target]);
  const laserScale = useMemo(() => laserRamp(laserAll), [laserAll]);
  return (
    // Reserve a stable height so the panel doesn't shimmy when the
    // histogram briefly drops out during a frame switch (the image unloads
    // until the next decode). The histogram block always mounts; its bins go
    // empty during the gap and re-fill once decode lands.
    <div className="flex min-h-[3.25rem] flex-wrap items-center gap-x-4 gap-y-2">
      {laser ? (
        <LaserResidualLegend
          residuals={laserAll.filter(inFrame)}
          ramp={laserScale}
          hist={residuals.laser_hist_per_cam?.[camera]}
        />
      ) : (
        run && (
          <ResidualLegend
            run={run}
            camera={frame.camera}
            view={frame.pose}
            ramp={targetScale}
          />
        )
      )}
      {frame.abs_path !== null && (
        <>
          <div className="flex items-center gap-2">
            <span className="font-mono text-[10px] uppercase tracking-wider text-fg-muted">
              histogram
            </span>
            <RoiHistogram
              counts={histogram ?? []}
              intensity={cursor?.intensity ?? null}
            />
          </div>
          <CursorChip cursor={cursor} />
        </>
      )}
    </div>
  );
}
