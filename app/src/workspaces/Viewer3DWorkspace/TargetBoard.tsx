import { useMemo } from "react";
import { iso3FromWire } from "../../lib/se3";
import type { Iso3Wire } from "../../store/types";
import type { TargetFeatureResidual } from "../../types";

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

  return (
    <group matrix={matrix} matrixAutoUpdate={false} onClick={onSelect}>
      <mesh>
        <planeGeometry
          args={[bbox.x1 - bbox.x0, bbox.y1 - bbox.y0]}
        />
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
            args={[
              new Float32Array(outlinePoints.flat()),
              3,
            ]}
          />
        </bufferGeometry>
        <lineBasicMaterial
          color={color}
          transparent
          opacity={ghost ? 0.25 : 1}
        />
      </line>
    </group>
  );
}

interface Bbox2 {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

function computeBoardBbox(residuals: TargetFeatureResidual[]): Bbox2 {
  if (residuals.length === 0) {
    // Fallback when residuals don't carry geometry — a 100 mm × 100 mm
    // square centred on the target origin.
    return { x0: -0.05, y0: -0.05, x1: 0.05, y1: 0.05 };
  }
  let x0 = Infinity;
  let y0 = Infinity;
  let x1 = -Infinity;
  let y1 = -Infinity;
  for (const r of residuals) {
    const [x, y] = r.target_xyz_m;
    if (x < x0) x0 = x;
    if (y < y0) y0 = y;
    if (x > x1) x1 = x;
    if (y > y1) y1 = y;
  }
  // Pad by 5 % to keep the marker dots from sitting on the outline.
  const pad = Math.max(x1 - x0, y1 - y0) * 0.05;
  return { x0: x0 - pad, y0: y0 - pad, x1: x1 + pad, y1: y1 + pad };
}
