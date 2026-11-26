import { TargetBoard as TargetBoardObject, disposeObject } from "@vitavision/three";
import { useEffect, useMemo } from "react";
import { iso3FromWire } from "../../lib/se3";
import type { Iso3Wire } from "../../store/types";
import type { TargetFeatureResidual } from "../../types";
import { computeBoardBbox } from "./boardGeometry";

interface TargetBoardProps {
  /** `rig_se3_target` (T_R_T) for the pose this board represents. */
  rigSe3Target: Iso3Wire;
  /** Pose-filtered residual records for this view. Used to size the
   * board from the actual target_xyz_m bounding box; fallback to a
   * 100 mm × 100 mm square when residuals are empty. */
  residuals: TargetFeatureResidual[];
  /** Outline color. */
  color: string;
  /** Translucent fill color for the plane. */
  fillColor: string;
  /** Lower opacity ghost mode for "show all poses". */
  ghost?: boolean;
  onSelect?: () => void;
}

/** `@vitavision/three`'s `TargetBoard`, sized to the bounding box of the
 * per-pose `target_xyz_m` residual records and placed by `rig_se3_target`
 * so it sits at the right place in the rig frame. Drawn translucent
 * (`setOpacity`); the non-ghost board is `setActive`, so its outline
 * stays visible through other geometry. Its outline is never pickable:
 * picks land on the surface. */
export function TargetBoard({
  rigSe3Target,
  residuals,
  color,
  fillColor,
  ghost = false,
  onSelect,
}: TargetBoardProps) {
  const matrix = useMemo(() => iso3FromWire(rigSe3Target), [rigSe3Target]);
  const bbox = useMemo(() => computeBoardBbox(residuals), [residuals]);
  const width = bbox.x1 - bbox.x0;
  const height = bbox.y1 - bbox.y0;

  const board = useMemo(
    () => new TargetBoardObject({ width, height, color: "gray", edgeColor: "gray" }),
    [width, height],
  );
  useEffect(() => () => disposeObject(board), [board]);
  useEffect(() => {
    board.setColors(fillColor, color);
    board.setOpacity(ghost ? 0.04 : 0.18);
    board.setActive(!ghost);
  }, [board, fillColor, color, ghost]);

  // The package board is centred on its origin, but target_xyz_m starts
  // at the board origin (often a corner), so shift it onto the bbox.
  const cx = (bbox.x0 + bbox.x1) / 2;
  const cy = (bbox.y0 + bbox.y1) / 2;

  return (
    <group
      matrix={matrix}
      matrixAutoUpdate={false}
      {...(onSelect !== undefined ? { onClick: onSelect } : {})}
    >
      <primitive object={board} position={[cx, cy, 0]} />
    </group>
  );
}
