/** Adapt a loaded export into the rig-shaped view the 3D scene renders
 *.
 *
 * The 3D workspace draws `cameras` + `cam_se3_rig` + `rig_se3_target`
 * (+ `laser_planes_rig`). Rig exports carry those directly. A
 * single-camera `laserline_device` export instead carries its laser
 * plane and per-view target poses under `estimate.params`, in the
 * camera's own frame. Placing that one camera at the rig origin
 * (identity `cam_se3_rig`) makes camera frame ≡ rig frame, so the laser
 * plane and `camera_se3_target` poses drop straight into the rig fields
 * and the existing `Scene` renders them unchanged.
 *
 * Pure and side-effect free so it unit-tests without React/Tauri. */
import type { AnyExport, Iso3Wire, PinholeCameraWire } from "../store/types";
import type { LaserlineDeviceExport } from "../types/generated/diagnose-wire";

const IDENTITY_ISO3: Iso3Wire = { rotation: [0, 0, 0, 1], translation: [0, 0, 0] };

/** True when the export already carries the rig triple the scene needs. */
function isRigShaped(data: AnyExport): boolean {
  return (
    Array.isArray(data.cameras) &&
    Array.isArray(data.cam_se3_rig) &&
    Array.isArray(data.rig_se3_target) &&
    data.cameras.length > 0
  );
}

/** Return a scene-renderable view of `data`.
 *
 * - Rig-shaped exports pass through untouched.
 * - A single-camera `laserline_device` export with a laser plane + poses
 *   is lifted into a one-camera rig at the origin (see module docs).
 * - Anything else is returned unchanged (the scene will show its empty
 *   state — e.g. planar / scheimpflug / single-cam hand-eye, which carry
 *   no extrinsic to place a frustum). */
export function adaptSceneExport(data: AnyExport): AnyExport {
  if (isRigShaped(data)) return data;
  if (data.kind !== "laserline_device") return data;

  const params = (data as unknown as LaserlineDeviceExport).estimate?.params;
  const plane = params?.plane;
  const poses = params?.poses;
  // Need both the plane (camera frame) and the per-view poses to build a
  // meaningful scene; bail (unchanged) if the export predates them.
  if (!plane || !params || !Array.isArray(poses) || poses.length === 0) {
    return data;
  }

  const camera: PinholeCameraWire = {
    k: params.intrinsics,
    dist: params.distortion,
    proj: null,
    sensor: null,
    _phantom: null,
  };

  return {
    ...data,
    cameras: [camera],
    cam_se3_rig: [IDENTITY_ISO3],
    // camera ≡ rig, so camera_se3_target poses are rig_se3_target.
    rig_se3_target: poses,
    // Plane is already in the camera (== rig) frame.
    laser_planes_rig: [plane],
    laser_planes_cam: [plane],
  };
}
