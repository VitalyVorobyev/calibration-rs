import { LaserFan } from "@vitavision/three-react";
import type { Matrix4 } from "three";
import { LASER_FAN_HALF_ANGLE } from "./laserFanPose";

interface LaserPlaneProps {
  /** `rig_se3_laser` from `laserFanPose`. */
  matrix: Matrix4;
  /** Fan reach in meters. */
  reach: number;
  /** Emphasised when its owning camera is the active one. */
  active?: boolean;
  onSelect?: () => void;
}

/** One calibrated laser plane, drawn as `@vitavision/three-react`'s
 * `LaserFan` light sheet (the scene's `defect` token, its outline not
 * pickable) in the frame `laserFanPose` picks for it. */
export function LaserPlane({ matrix, reach, active = false, onSelect }: LaserPlaneProps) {
  return (
    <group matrix={matrix} matrixAutoUpdate={false}>
      <LaserFan
        halfAngle={LASER_FAN_HALF_ANGLE}
        length={reach}
        active={active}
        onSelect={onSelect}
      />
    </group>
  );
}
