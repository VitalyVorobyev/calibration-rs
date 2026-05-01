import { Matrix4, Quaternion, Vector3 } from "three";
import type { Iso3Wire } from "../store/types";

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
