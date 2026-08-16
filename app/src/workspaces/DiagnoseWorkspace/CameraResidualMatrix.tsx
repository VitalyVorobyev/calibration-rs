import { useMemo } from "react";
import { colorForError } from "../../components/FrameCanvas";
import { Table, Td, Th } from "../../components/ui";
import { computeCameraPoseMatrix } from "../../lib/residualStats";
import type { TargetFeatureResidual } from "../../types";

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
        <Table size="xs">
          <thead>
            <tr>
              <Th sticky className="p-1 text-muted-foreground">
                cam\pose
              </Th>
              {matrix.poses.map((pose) => (
                <Th
                  key={pose}
                  align="center"
                  className={
                    pose === selectedPose ? "p-1 text-brand" : "p-1 text-muted-foreground"
                  }
                >
                  {pose}
                </Th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.cameras.map((camera) => (
              <tr key={camera}>
                <Th
                  scope="row"
                  sticky
                  className={
                    camera === selectedCamera
                      ? "p-1 text-brand"
                      : "p-1 text-muted-foreground"
                  }
                >
                  {camera}
                </Th>
                {matrix.poses.map((pose) => {
                  const cell = matrix.cells.get(camera)?.get(pose);
                  const active = pose === selectedPose && camera === selectedCamera;
                  return (
                    <Td key={pose} className="p-0.5">
                      <MatrixCell
                        mean={cell?.mean ?? null}
                        count={cell?.count ?? 0}
                        pose={pose}
                        camera={camera}
                        active={active}
                        onSelect={onSelect}
                      />
                    </Td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </Table>
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
      className={`flex h-6 w-8 items-center justify-center rounded-[3px] text-[9px] transition-transform hover:scale-105 ${
        observed
          ? "border-none text-black/75"
          : "border border-dashed border-border text-muted-foreground"
      } ${active ? "ring-1 ring-brand" : ""}`}
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
    <div className="flex items-center gap-2 font-mono text-[10px] text-muted-foreground">
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
