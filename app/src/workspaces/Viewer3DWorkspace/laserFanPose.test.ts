import { Vector3 } from "three";
import { describe, expect, it } from "vitest";
import { laserFanPose } from "./laserFanPose";

const IDENTITY = {
  rotation: [0, 0, 0, 1] as [number, number, number, number],
  translation: [0, 0, 0] as [number, number, number],
};

function axes(m: { elements: ArrayLike<number> }) {
  const e = m.elements;
  return {
    x: new Vector3(e[0], e[1], e[2]),
    y: new Vector3(e[4], e[5], e[6]),
    z: new Vector3(e[8], e[9], e[10]),
    origin: new Vector3(e[12], e[13], e[14]),
  };
}

describe("laserFanPose", () => {
  it("roots the fan at the camera centre projected onto the plane, opening along the optical axis", () => {
    // Plane x = 0.05 (n = +X, d = -0.05); camera at the rig origin looking +Z.
    const { matrix, reach } = laserFanPose(
      { normal: [1, 0, 0], distance: -0.05 },
      IDENTITY,
    );
    const a = axes(matrix);
    expect(a.origin.toArray()).toEqual([0.05, 0, 0]);
    expect(a.x.toArray()).toEqual([1, 0, 0]);
    expect(a.z.x).toBeCloseTo(0, 12);
    expect(a.z.y).toBeCloseTo(0, 12);
    expect(a.z.z).toBeCloseTo(1, 12);
    expect(a.x.clone().cross(a.y).distanceTo(a.z)).toBeLessThan(1e-12);
    expect(reach).toBeCloseTo(0.15, 12);
  });

  it("keeps the fan in the plane for a tilted plane and a moved camera", () => {
    const n = new Vector3(1, 0.2, -0.4).normalize();
    const plane = { normal: n.toArray(), distance: -0.1 };
    // cam_se3_rig of a camera at (0.2, 0.1, 0): translation = -R^T c with R = I.
    const cam = {
      rotation: IDENTITY.rotation,
      translation: [-0.2, -0.1, 0] as [number, number, number],
    };
    const a = axes(laserFanPose(plane, cam).matrix);
    expect(n.dot(a.origin) + plane.distance).toBeCloseTo(0, 12);
    expect(a.z.dot(n)).toBeCloseTo(0, 12);
    expect(a.y.dot(n)).toBeCloseTo(0, 12);
  });

  it("falls back to the closest point to the origin without an owning camera", () => {
    const { matrix, reach } = laserFanPose({ normal: [0, 0, 1], distance: -0.3 });
    const a = axes(matrix);
    expect(a.origin.toArray()).toEqual([0, 0, 0.3]);
    expect(a.z.dot(new Vector3(0, 0, 1))).toBeCloseTo(0, 12);
    expect(reach).toBe(0.3);
  });
});
