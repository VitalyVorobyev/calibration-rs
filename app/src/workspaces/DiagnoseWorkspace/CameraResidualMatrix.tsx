import { cn, focusRing } from "@vitavision/ui";
import { useMemo } from "react";
import { colorForError } from "../../lib/errorColors";
import { computeCameraPoseMatrix } from "../../lib/residualStats";
import type { TargetFeatureResidual } from "../../types";

/** Pins the camera column while the pose axis scrolls horizontally. */
const STICKY = "sticky left-0 z-10 bg-surface";

interface CameraResidualMatrixProps {
  residuals: TargetFeatureResidual[];
  selectedPose: number;
  selectedCamera: number;
  onSelect: (pose: number, camera: number) => void;
}

/** Cameras × poses grid coloured by mean reprojection residual, reusing
 * the FrameCanvas severity scale. Only rendered for multi-camera
 * exports; click a cell to jump the viewer to that `(pose, camera)`
 *. Tooltip carries the exact numbers. */
export function CameraResidualMatrix({
  residuals,
  selectedPose,
  selectedCamera,
  onSelect,
}: CameraResidualMatrixProps) {
  const matrix = useMemo(() => computeCameraPoseMatrix(residuals), [residuals]);

  if (matrix.cameras.length < 2 || matrix.poses.length === 0) {
    // Single-camera exports get nothing useful from a 1-row matrix; the
    // per-pose table already covers them.
    return null;
  }

  return (
    <section aria-label="Cross-camera residual matrix" className="flex flex-col gap-1.5">
      <div className="overflow-x-auto">
        {/* A raw <table>, not ui's data-driven `Table`: this grid needs a
            sticky row-header column and per-cell buttons. */}
        <table className="border-collapse font-mono text-[10px] tabular-nums">
          <thead>
            <tr>
              <th
                scope="col"
                className={cn(STICKY, "p-1 text-left font-medium text-fg-muted")}
              >
                cam\pose
              </th>
              {matrix.poses.map((pose) => (
                <th
                  key={pose}
                  scope="col"
                  className={cn(
                    "p-1 text-center font-medium",
                    pose === selectedPose ? "text-signal" : "text-fg-muted",
                  )}
                >
                  {pose}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.cameras.map((camera) => (
              <tr key={camera}>
                <th
                  scope="row"
                  className={cn(
                    STICKY,
                    "p-1 text-left font-medium",
                    camera === selectedCamera ? "text-signal" : "text-fg-muted",
                  )}
                >
                  {camera}
                </th>
                {matrix.poses.map((pose) => {
                  const cell = matrix.cells.get(camera)?.get(pose);
                  const active = pose === selectedPose && camera === selectedCamera;
                  return (
                    <td key={pose} className="p-0.5">
                      <MatrixCell
                        mean={cell?.mean ?? null}
                        count={cell?.count ?? 0}
                        pose={pose}
                        camera={camera}
                        active={active}
                        onSelect={onSelect}
                      />
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <MatrixLegend />
    </section>
  );
}

interface MatrixCellProps {
  mean: number | null;
  count: number;
  pose: number;
  camera: number;
  active: boolean;
  onSelect: (pose: number, camera: number) => void;
}

function MatrixCell({ mean, count, pose, camera, active, onSelect }: MatrixCellProps) {
  const observed = mean != null;
  const title = observed
    ? `cam ${camera} · pose ${pose}: mean ${mean.toFixed(3)} px over ${count} corner${
        count !== 1 ? "s" : ""
      }`
    : `cam ${camera} · pose ${pose}: no residuals`;
  return (
    <button
      type="button"
      onClick={() => onSelect(pose, camera)}
      title={title}
      aria-label={title}
      className={cn(
        "flex h-6 w-8 items-center justify-center rounded-[3px] text-[9px] transition-transform hover:scale-105",
        observed
          ? "border-none text-black/75"
          : "border border-dashed border-line text-fg-muted",
        active && "ring-1 ring-signal",
        focusRing,
      )}
      style={{ background: observed ? colorForError(mean) : "transparent" }}
    >
      {observed ? mean.toFixed(1) : "·"}
    </button>
  );
}

function MatrixLegend() {
  const swatches: { err: number; label: string }[] = [
    { err: 0.5, label: "<1" },
    { err: 1.5, label: "<2" },
    { err: 3, label: "<5" },
    { err: 7, label: "<10" },
    { err: 20, label: "≥10" },
  ];
  return (
    <div className="flex items-center gap-2 font-mono text-[10px] text-fg-muted">
      <span>px</span>
      {swatches.map((s) => (
        <span key={s.label} className="inline-flex items-center gap-1">
          <span
            className="inline-block h-2 w-2 rounded-[2px]"
            style={{ background: colorForError(s.err) }}
            aria-hidden="true"
          />
          {s.label}
        </span>
      ))}
    </div>
  );
}
