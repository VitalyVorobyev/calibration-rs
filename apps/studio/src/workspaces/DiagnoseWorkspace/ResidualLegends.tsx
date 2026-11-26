import { colorForError, colorForLaserError } from "../../lib/errorColors";
import type {
  CursorReadout,
  FeatureResidualHistogram,
  LaserFeatureResidual,
  TargetFeatureResidual,
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

/** The frame's target residual statistics and the colour ramp key. */
export function ResidualLegend({ residuals }: { residuals: TargetFeatureResidual[] }) {
  const errs = residuals
    .map((r) => r.error_px)
    .filter((e): e is number => typeof e === "number");
  const mean = errs.length > 0 ? errs.reduce((a, b) => a + b, 0) / errs.length : 0;
  const max = errs.length > 0 ? Math.max(...errs) : 0;
  const diverged = residuals.length - errs.length;
  return (
    <div className="flex flex-wrap items-center gap-x-3.5 gap-y-1 font-mono text-[11px] text-fg-muted">
      <span>features {residuals.length}</span>
      <span>diverged {diverged}</span>
      <span>mean {mean.toFixed(3)} px</span>
      <span>max {max.toFixed(3)} px</span>
      <span className="flex items-center gap-2">
        <Swatch color={colorForError(0.5)} label="<1" />
        <Swatch color={colorForError(1.5)} label="<2" />
        <Swatch color={colorForError(3)} label="<5" />
        <Swatch color={colorForError(7)} label="<10" />
        <Swatch color={colorForError(20)} label="≥10" />
      </span>
    </div>
  );
}

/** The frame's laser point-to-plane statistics and the colour ramp key. */
export function LaserResidualLegend({
  residuals,
  hist,
}: {
  residuals: LaserFeatureResidual[];
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
      <span className="flex items-center gap-2">
        <Swatch color={colorForLaserError(0.1)} label="<0.2" />
        <Swatch color={colorForLaserError(0.3)} label="<0.5" />
        <Swatch color={colorForLaserError(0.7)} label="<1" />
        <Swatch color={colorForLaserError(1.5)} label="<2" />
        <Swatch color={colorForLaserError(3)} label="≥2" />
      </span>
    </div>
  );
}

function Swatch({ color, label }: { color: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1">
      <span
        className="inline-block h-2 w-2 rounded-[2px]"
        style={{ background: color }}
      />
      {label}
    </span>
  );
}
