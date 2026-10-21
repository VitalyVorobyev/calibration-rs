import { describe, expect, it } from "vitest";
import type { AnyExport } from "../store/types";
import { adaptSceneExport } from "./sceneExport";

const K = { fx: 1000, fy: 1000, cx: 320, cy: 240 };
const DIST = { k1: 0, k2: 0, k3: 0, p1: 0, p2: 0, iters: 5 };
const CAMERA = {
  projection: { type: "pinhole" },
  distortion: { type: "brown_conrady5", ...DIST },
  sensor: { type: "identity" },
  intrinsics: K,
};

const TILTED_CAMERA = {
  ...CAMERA,
  sensor: { type: "scheimpflug", tilt_x: 0.02, tilt_y: -0.01 },
};

const POSE = {
  rotation: [0, 0, 0, 1] as [number, number, number, number],
  translation: [0, 0, 0.5] as [number, number, number],
};
const PLANE = { normal: [0, 1, 0] as [number, number, number], distance: -0.1 };

describe("adaptSceneExport", () => {
  it("passes rig-shaped exports through unchanged", () => {
    const rig = {
      kind: "rig_extrinsics",
      per_feature_residuals: { target: [] },
      cameras: [CAMERA],
      cam_se3_rig: [POSE],
      rig_se3_target: [POSE],
    } as unknown as AnyExport;
    expect(adaptSceneExport(rig)).toBe(rig);
  });

  it("lifts a single-cam laserline export into a one-camera rig at the origin", () => {
    const laser = {
      kind: "laserline_device",
      per_feature_residuals: { target: [], laser: [] },
      estimate: {
        params: {
          camera: TILTED_CAMERA,
          plane: PLANE,
          poses: [POSE, POSE],
        },
      },
    } as unknown as AnyExport;

    const adapted = adaptSceneExport(laser);
    expect(adapted.cameras).toHaveLength(1);
    // The camera keeps its model (here a tilted sensor).
    expect(adapted.cameras).toEqual([TILTED_CAMERA]);
    // Camera pinned at the rig origin (identity extrinsic).
    expect(adapted.cam_se3_rig).toEqual([
      { rotation: [0, 0, 0, 1], translation: [0, 0, 0] },
    ]);
    // camera_se3_target poses become rig_se3_target.
    expect(adapted.rig_se3_target).toHaveLength(2);
    // Plane appears in both rig and camera frame (they coincide).
    expect(adapted.laser_planes_rig).toEqual([PLANE]);
    expect(adapted.laser_planes_cam).toEqual([PLANE]);
  });

  it("leaves a laserline export without plane/poses unchanged", () => {
    const bare = {
      kind: "laserline_device",
      per_feature_residuals: { target: [] },
      estimate: { params: { camera: CAMERA } },
    } as unknown as AnyExport;
    const adapted = adaptSceneExport(bare);
    expect(adapted.cameras).toBeUndefined();
    expect(adapted).toBe(bare);
  });

  it("leaves other single-camera exports (planar) unchanged", () => {
    const planar = {
      kind: "planar_intrinsics",
      per_feature_residuals: { target: [] },
    } as unknown as AnyExport;
    expect(adaptSceneExport(planar)).toBe(planar);
  });
});
