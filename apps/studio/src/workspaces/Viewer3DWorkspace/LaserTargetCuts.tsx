import { useMemo } from "react";
import { iso3FromWire } from "../../lib/se3";
import type { Iso3Wire, LaserPlaneWire } from "../../store/types";
import type { TargetFeatureResidual } from "../../types";
import {
  computeBoardBbox,
  intersectLaserPlaneWithTarget,
  type Segment2,
} from "./boardGeometry";

interface LaserTargetCutsProps {
  /** Active pose `rig_se3_target` (T_R_T). */
  rigSe3Target: Iso3Wire;
  /** Active-pose target residuals, used to clip cuts to the board bbox. */
  residuals: TargetFeatureResidual[];
  /** Laser planes in rig frame. */
  planesRig: LaserPlaneWire[];
}

const CUT_COLORS = ["#38bdf8", "#f97316", "#a3e635", "#f43f5e", "#c084fc", "#facc15"];

export function LaserTargetCuts({
  rigSe3Target,
  residuals,
  planesRig,
}: LaserTargetCutsProps) {
  const matrix = useMemo(() => iso3FromWire(rigSe3Target), [rigSe3Target]);
  const bbox = useMemo(() => computeBoardBbox(residuals), [residuals]);
  const segments = useMemo(
    () =>
      planesRig
        .map((plane, idx) => ({
          idx,
          segment: intersectLaserPlaneWithTarget(plane, rigSe3Target, bbox),
        }))
        .filter((v): v is { idx: number; segment: Segment2 } => v.segment != null),
    [planesRig, rigSe3Target, bbox],
  );

  if (segments.length === 0) return null;

  return (
    <group matrix={matrix} matrixAutoUpdate={false}>
      {segments.map(({ idx, segment }) => (
        <line key={`laser-cut-${idx}`}>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              args={[
                new Float32Array([
                  segment.a[0],
                  segment.a[1],
                  0.0004,
                  segment.b[0],
                  segment.b[1],
                  0.0004,
                ]),
                3,
              ]}
            />
          </bufferGeometry>
          <lineBasicMaterial
            color={CUT_COLORS[idx % CUT_COLORS.length]!}
            transparent
            opacity={0.95}
          />
        </line>
      ))}
    </group>
  );
}
