import { Euler, Matrix4, Quaternion, Vector3 } from "three";
import type { Iso3Wire } from "../store/types";

const RAD2DEG = 180 / Math.PI;
export const IDENTITY_ISO3: Iso3Wire = {
  rotation: [0, 0, 0, 1],
  translation: [0, 0, 0],
};

/** Build a Three.js Matrix4 from the on-wire `Iso3` shape.
 *
 * `nalgebra::Isometry3` serializes rotation as `[qx, qy, qz, qw]` (the
 * imaginary parts first, scalar last) and translation as `[tx, ty, tz]`,
 * matching Three.js's `Quaternion(x, y, z, w)` constructor. */
export function iso3FromWire(iso: Iso3Wire): Matrix4 {
  const [qx, qy, qz, qw] = iso.rotation;
  const [tx, ty, tz] = iso.translation;
  const m = new Matrix4();
  const q = new Quaternion(qx, qy, qz, qw);
  m.compose(new Vector3(tx, ty, tz), q, new Vector3(1, 1, 1));
  return m;
}

/** Inverted form. `cam_se3_rig` is T_C_R (rig→camera); placing the
 * camera object in the scene needs T_R_C (camera position in rig
 * frame), i.e. the inverse. Computing the inverse from the wire fields
 * directly avoids a Matrix4 round-trip and the floating-point drift
 * that comes with it. */
export function iso3InverseFromWire(iso: Iso3Wire): Matrix4 {
  const [qx, qy, qz, qw] = iso.rotation;
  const [tx, ty, tz] = iso.translation;
  // Inverse of (R, t) is (R^T, -R^T t). For unit quaternions R^T == conj(R).
  const qInv = new Quaternion(-qx, -qy, -qz, qw);
  const tNeg = new Vector3(-tx, -ty, -tz).applyQuaternion(qInv);
  return new Matrix4().compose(tNeg, qInv, new Vector3(1, 1, 1));
}

/** Compose two SE(3) wire transforms: result = a · b (matrix multiply,
 * not a kinematic chain rename — caller picks the convention). */
export function iso3Compose(a: Iso3Wire, b: Iso3Wire): Iso3Wire {
  const [aqx, aqy, aqz, aqw] = a.rotation;
  const [atx, aty, atz] = a.translation;
  const [bqx, bqy, bqz, bqw] = b.rotation;
  const [btx, bty, btz] = b.translation;
  const aq = new Quaternion(aqx, aqy, aqz, aqw);
  const bq = new Quaternion(bqx, bqy, bqz, bqw);
  const cq = aq.clone().multiply(bq);
  const bt = new Vector3(btx, bty, btz).applyQuaternion(aq);
  return {
    rotation: [cq.x, cq.y, cq.z, cq.w],
    translation: [atx + bt.x, aty + bt.y, atz + bt.z],
  };
}

/** Extract the camera-position in world frame from `cam_se3_rig` (T_C_R).
 * Used to seed OrbitControls auto-fit and the scene bounding box. */
export function cameraPositionInRig(camSe3Rig: Iso3Wire): [number, number, number] {
  const [qx, qy, qz, qw] = camSe3Rig.rotation;
  const [tx, ty, tz] = camSe3Rig.translation;
  // pos = -R^T t = -(conj(q) * t * q-rotated)
  const qInv = new Quaternion(-qx, -qy, -qz, qw);
  const tNeg = new Vector3(-tx, -ty, -tz).applyQuaternion(qInv);
  return [tNeg.x, tNeg.y, tNeg.z];
}

/** SE(3) inverse in wire format. Same math as `iso3InverseFromWire`
 * but stays in `Iso3Wire` so it can be composed with the other
 * pose-readout helpers without a Matrix4 round-trip. */
export function iso3InverseWire(iso: Iso3Wire): Iso3Wire {
  const [qx, qy, qz, qw] = iso.rotation;
  const [tx, ty, tz] = iso.translation;
  const qInv = new Quaternion(-qx, -qy, -qz, qw);
  const tNeg = new Vector3(-tx, -ty, -tz).applyQuaternion(qInv);
  return {
    rotation: [qInv.x, qInv.y, qInv.z, qInv.w],
    translation: [tNeg.x, tNeg.y, tNeg.z],
  };
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
