import { composeIso3, invertIso3, matrixFromIso3 } from "@vitavision/three";
import { Euler, type Matrix4, Quaternion } from "three";
import type { Iso3Wire } from "../store/types";

const RAD2DEG = 180 / Math.PI;
export const IDENTITY_ISO3: Iso3Wire = {
  rotation: [0, 0, 0, 1],
  translation: [0, 0, 0],
};

// The SE(3) algebra (wire → matrix, inverse, compose) is
// `@vitavision/three`'s: one copy of the nalgebra `Isometry3` wire
// convention (`[qx, qy, qz, qw]`, scalar last) shared with the other
// vitavision viewers. The helpers below keep this app's names.

/** Build a Three.js Matrix4 from the on-wire `Iso3` shape. */
export function iso3FromWire(iso: Iso3Wire): Matrix4 {
  return matrixFromIso3(iso);
}

/** Inverted form. `cam_se3_rig` is T_C_R (rig→camera); placing the
 * camera object in the scene needs T_R_C (camera position in rig
 * frame), i.e. the inverse. Inverted on the wire fields, so there is no
 * Matrix4 round-trip. */
export function iso3InverseFromWire(iso: Iso3Wire): Matrix4 {
  return matrixFromIso3(invertIso3(iso));
}

/** Compose two SE(3) wire transforms: result = a · b (matrix multiply,
 * not a kinematic chain rename — caller picks the convention). */
export function iso3Compose(a: Iso3Wire, b: Iso3Wire): Iso3Wire {
  return composeIso3(a, b);
}

/** Extract the camera-position in world frame from `cam_se3_rig` (T_C_R).
 * Used to seed OrbitControls auto-fit and the scene bounding box. */
export function cameraPositionInRig(camSe3Rig: Iso3Wire): [number, number, number] {
  return invertIso3(camSe3Rig).translation;
}

/** SE(3) inverse in wire format, so it composes with the other
 * pose-readout helpers without a Matrix4 round-trip. */
export function iso3InverseWire(iso: Iso3Wire): Iso3Wire {
  return invertIso3(iso);
}

/** Euclidean magnitude of the translation, in the same units as the
 * input (meters for our wire format). Used for "distance" readouts in
 * the 3D viewer pose panels. */
export function iso3DistanceM(iso: Iso3Wire): number {
  const [tx, ty, tz] = iso.translation;
  return Math.hypot(tx, ty, tz);
}

/** Convert the rotation to ZYX Euler angles in degrees.
 *
 * Three.js's `Euler.setFromQuaternion` with order `"XYZ"` returns the
 * angles such that `R = Rx(x) * Ry(y) * Rz(z)`. For a board pose the
 * X / Y components are the pitch / yaw the engineer reads as "is the
 * board tilted left, is it tilted up"; Z is roll. */
export function iso3EulerXYZDeg(iso: Iso3Wire): {
  x: number;
  y: number;
  z: number;
} {
  const [qx, qy, qz, qw] = iso.rotation;
  const q = new Quaternion(qx, qy, qz, qw);
  const e = new Euler().setFromQuaternion(q, "XYZ");
  return { x: e.x * RAD2DEG, y: e.y * RAD2DEG, z: e.z * RAD2DEG };
}

/** Magnitude of the axis-angle representation of the rotation, in
 * degrees. A single number summarising "how rotated is this pose". */
export function iso3RotationAngleDeg(iso: Iso3Wire): number {
  const qw = Math.min(1, Math.max(-1, iso.rotation[3]));
  return 2 * Math.acos(qw) * RAD2DEG;
}

/** Compose `cam_se3_rig[ref]` with `cam_se3_rig[sel]^-1` to get the
 * pose of the selected camera expressed in the reference camera's
 * frame: translation tells you where the selected camera sits relative
 * to the reference, rotation tells you how its axes are oriented. */
export function relativeCameraPose(
  refCamSe3Rig: Iso3Wire,
  selCamSe3Rig: Iso3Wire,
): Iso3Wire {
  return iso3Compose(refCamSe3Rig, iso3InverseWire(selCamSe3Rig));
}

/** Compose `cam_se3_rig[cam] · rig_se3_target[pose]` to get the
 * target's pose in the camera's frame. The translation is the
 * camera→target vector (length = distance to board); the rotation is
 * the board's orientation in the camera. */
export function targetInCameraPose(
  camSe3Rig: Iso3Wire,
  rigSe3Target: Iso3Wire,
): Iso3Wire {
  return iso3Compose(camSe3Rig, rigSe3Target);
}
