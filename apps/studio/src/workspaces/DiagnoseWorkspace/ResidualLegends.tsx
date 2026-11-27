import { api } from "../../api";
import type { RunRef } from "../../api/generated/types";
import { ARROW_GAIN } from "../../components/frameOverlayPaths";
import { ColormapLegend } from "@vitavision/charts";
import type { ErrorRamp } from "../../lib/errorColors";
import { useQuery } from "../../session/query";
import type {
  CursorReadout,
  FeatureResidualHistogram,
  LaserFeatureResidual,
} from "../../types";

/** The cursor's image position and luminance. */
export function CursorChip({ cursor }: { cursor: CursorReadout | null }) {
  return (
    <span className="inline-flex min-w-[12rem] items-center gap-2 rounded-control border border-line bg-surface px-2 py-1 font-mono text-[11px] text-fg-muted">
      <span className="uppercase tracking-wider">cursor</span>
      {cursor ? (
        <span className="text-fg tabular-nums">
          ({cursor.x.toFixed(0)}, {cursor.y.toFixed(0)})
          {cursor.intensity != null ? ` · I=${cursor.intensity}` : ""}
        </span>
      ) : (
        <span>—</span>
      )}
    </span>
  );
}

/** The frame's target residual statistics (the back end's, of camera
 * `camera` in view `view` of `run`) and the colour scale of its arrows. */
export function ResidualLegend({
  run,
  camera,
  view,
  ramp,
}: {
  run: RunRef;
  camera: number;
  view: number;
  /** The scale the arrows are drawn on (one for the whole run). */
  ramp: ErrorRamp;
}) {
  const q = useQuery("residual_stats", { run }, () => api.residualStats({ run }));
  const cell = q.data?.cells.find((c) => c.camera === camera && c.view === view);
  const px = (v: number | undefined) => (v === undefined ? "–" : `${v.toFixed(3)} px`);
  // A frame without records has none in the statistics.
  const empty = q.data !== undefined && cell === undefined;
  return (
    <div className="flex flex-wrap items-center gap-x-3.5 gap-y-1 font-mono text-[11px] text-fg-muted">
      <span>features {cell ? cell.stats.count + cell.unprojected : empty ? 0 : "–"}</span>
      <span>diverged {cell ? cell.unprojected : empty ? 0 : "–"}</span>
      <span>mean {px(cell?.stats.mean)}</span>
      <span>max {px(cell?.stats.max)}</span>
      <span>arrows: observed − projected, from the observed corner, ×{ARROW_GAIN}</span>
      <RampLegend ramp={ramp} label="Reprojection error" />
    </div>
  );
}

/** The frame's laser point-to-plane statistics and the colour ramp key. */
export function LaserResidualLegend({
  residuals,
  hist,
  ramp,
}: {
  residuals: LaserFeatureResidual[];
  /** The scale the laser points are drawn on (one for the whole run). */
  ramp: ErrorRamp;
  hist?: FeatureResidualHistogram | null | undefined;
}) {
  const mm = residuals
    .map((r) => r.residual_m)
    .filter((d): d is number => typeof d === "number")
    .map((d) => Math.abs(d) * 1e3);
  const mean = mm.length > 0 ? mm.reduce((a, b) => a + b, 0) / mm.length : 0;
  const max = mm.length > 0 ? Math.max(...mm) : 0;
  const missed = residuals.length - mm.length;
  return (
    <div className="flex flex-wrap items-center gap-x-3.5 gap-y-1 font-mono text-[11px] text-fg-muted">
      <span>laser pts {residuals.length}</span>
      <span>no-intersect {missed}</span>
      <span>mean {mean.toFixed(3)} mm</span>
      <span>max {max.toFixed(3)} mm</span>
      {hist && <span>cam px-mean {hist.mean.toFixed(3)}</span>}
      <RampLegend ramp={ramp} label="Point-to-plane distance" />
    </div>
  );
}

/** The colour scale an overlay is drawn on: its range, unit and whether some
 * values lie above it. */
function RampLegend({ ramp, label }: { ramp: ErrorRamp; label: string }) {
  return (
    <ColormapLegend
      map={ramp.map}
      domain={ramp.domain}
      unit={ramp.unit}
      label={label}
      clipped={ramp.clipped ? "high" : undefined}
      className="shrink-0 whitespace-nowrap"
    />
  );
}
