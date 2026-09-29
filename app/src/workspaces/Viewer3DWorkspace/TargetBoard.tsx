import { useMemo } from "react";
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

/** Translucent plane mesh sized to the bounding box of the per-pose
 * `target_xyz_m` residual records, transformed by `rig_se3_target` so
 * it sits at the right place in the rig frame. */
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

  // Outline: the four corners as a closed line loop. Drawn separately
  // from the filled mesh so the wire frame stays sharp under any
  // opacity setting.
  const outlinePoints: [number, number, number][] = [
    [bbox.x0, bbox.y0, 0],
    [bbox.x1, bbox.y0, 0],
    [bbox.x1, bbox.y1, 0],
    [bbox.x0, bbox.y1, 0],
    [bbox.x0, bbox.y0, 0],
  ];

  // `planeGeometry` is centered at the local origin, but the outline
  // uses absolute board coordinates (target_xyz_m starts at the board
  // origin, often a corner — not symmetric around zero). Translate the
  // mesh so the centered plane aligns with the absolute outline,
  // otherwise fill and outline are visibly offset by half the bbox.
  const cx = (bbox.x0 + bbox.x1) / 2;
  const cy = (bbox.y0 + bbox.y1) / 2;

  return (
    <group
      matrix={matrix}
      matrixAutoUpdate={false}
      {...(onSelect !== undefined ? { onClick: onSelect } : {})}
    >
      <mesh position={[cx, cy, 0]}>
        <planeGeometry args={[bbox.x1 - bbox.x0, bbox.y1 - bbox.y0]} />
        <meshBasicMaterial
          color={fillColor}
          transparent
          opacity={ghost ? 0.04 : 0.18}
          depthWrite={false}
        />
      </mesh>
      <line>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[new Float32Array(outlinePoints.flat()), 3]}
          />
        </bufferGeometry>
        <lineBasicMaterial color={color} transparent opacity={ghost ? 0.25 : 1} />
      </line>
    </group>
  );
}
