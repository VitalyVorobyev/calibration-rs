/** Where the 3D viewer draws a calibrated laser plane.
 *
 * An export carries each laser as a plane (`n · p + d = 0`, rig frame),
 * not as an emitter pose. `@vitavision/three`'s `LaserFan` draws a light
 * sheet from an emitter: a sector in its frame's `x = 0` plane, opening
 * about +Z. This picks that frame for plane `i` from its owning camera
 * `i`: the apex is the camera centre projected onto the plane (the
 * emitter sits beside the camera on a laserline device), the fan opens
 * along the camera's optical axis projected into the plane (towards the
 * measurement volume), and it reaches out in proportion to the
 * camera↔plane distance — the natural scale of the device.
 *
 * Pure and side-effect free so it unit-tests without React. */
import { invertIso3 } from "@vitavision/three";
import { Matrix4, Quaternion, Vector3 } from "three";
import type { Iso3Wire, LaserPlaneWire } from "../../store/types";

/** Half-angle of the drawn fan (a 90° sheet). The export does not carry
 * the laser's real fan angle. */
export const LASER_FAN_HALF_ANGLE = Math.PI / 4;

const REACH_MIN_M = 0.1;
const REACH_MAX_M = 0.6;
const REACH_FALLBACK_M = 0.3;

export interface LaserFanPose {
  /** `rig_se3_laser`: X = plane normal, Z = the fan's central ray. */
  matrix: Matrix4;
  /** Fan reach along its central ray, in meters. */
  reach: number;
}

/** The fan frame and reach for `plane`, owned by the camera at
 * `camSe3Rig` (T_C_R) when there is one. */
export function laserFanPose(plane: LaserPlaneWire, camSe3Rig?: Iso3Wire): LaserFanPose {
  const n = new Vector3(...plane.normal).normalize();
  let apex: Vector3;
  let forward: Vector3;
  let reach: number;
  if (camSe3Rig) {
    const rigSe3Cam = invertIso3(camSe3Rig);
    const centre = new Vector3(...rigSe3Cam.translation);
    const signed = n.dot(centre) + plane.distance;
    apex = centre.addScaledVector(n, -signed);
    const axis = new Vector3(0, 0, 1).applyQuaternion(
      new Quaternion(...rigSe3Cam.rotation),
    );
    forward = axis.addScaledVector(n, -axis.dot(n));
    reach = Math.min(REACH_MAX_M, Math.max(REACH_MIN_M, 3 * Math.abs(signed)));
  } else {
    // No owning camera: the plane's closest point to the rig origin.
    apex = n.clone().multiplyScalar(-plane.distance);
    forward = new Vector3();
    reach = REACH_FALLBACK_M;
  }
  if (forward.lengthSq() < 1e-12) {
    // Optical axis along the normal (or no camera): any in-plane direction.
    forward = new Vector3(0, 0, 1).addScaledVector(n, -n.z);
    if (forward.lengthSq() < 1e-12) forward.set(1, 0, 0).addScaledVector(n, -n.x);
  }
  forward.normalize();
  // Right-handed: X × Y = Z with X = n, Z = forward.
  const y = forward.clone().cross(n);
  const matrix = new Matrix4().makeBasis(n, y, forward).setPosition(apex);
  return { matrix, reach };
}
