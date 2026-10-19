import { describe, expect, it } from "vitest";
import type { ParameterCovariance } from "../types/generated/diagnose-wire";
import type { AnyExport } from "../store/types";
import {
  correlationMatrix,
  displayUnit,
  exportUncertainty,
  parameterRows,
  unavailableText,
} from "./uncertainty";

const COV: ParameterCovariance = {
  labels: ["fx", "cam1.k1", "cam1.cam_se3_rig.rz", "cam1.cam_se3_rig.tx", "tilt_x"],
  covariance: [
    [4, 0, 0, 0, 0],
    [0, 1e-6, 0, 0, 0],
    [0, 0, 1e-6, 0, 0],
    [0, 0, 0, 1e-8, 0],
    [0, 0, 0, 0, 1e-6],
  ],
  noise: [{ kind: "reprojection", sigma: 0.1, count: 100 }],
  num_residuals: 100,
  dof: 95,
};

describe("displayUnit", () => {
  it("shows rotations in degrees, translations in mm, intrinsics in px", () => {
    expect(displayUnit("cam1.cam_se3_rig.rz").unit).toBe("°");
    expect(displayUnit("tilt_x").unit).toBe("°");
    expect(displayUnit("gripper_se3_rig.tx").unit).toBe("mm");
    expect(displayUnit("plane.d").unit).toBe("mm");
    expect(displayUnit("cam0.cx").unit).toBe("px");
    expect(displayUnit("k1").unit).toBe("");
    expect(displayUnit("plane.nx").unit).toBe("");
  });
});

describe("parameterRows", () => {
  it("converts value and σ to display units", () => {
    const rows = parameterRows(COV, [800, -0.1, null, null, 0.05]);
    expect(rows[0]).toEqual({ label: "fx", value: 800, sigma: 2, unit: "px" });
    expect(rows[1]!.value).toBe(-0.1);
    expect(rows[2]!.value).toBeNull();
    expect(rows[2]!.sigma).toBeCloseTo((1e-3 * 180) / Math.PI);
    expect(rows[3]!.sigma).toBeCloseTo(0.1);
    expect(rows[4]!.value).toBeCloseTo((0.05 * 180) / Math.PI);
  });

  it("leaves values empty when none are known", () => {
    expect(parameterRows(COV, []).every((r) => r.value === null)).toBe(true);
  });
});

describe("correlationMatrix", () => {
  it("normalizes the covariance and marks zero variances", () => {
    const rho = correlationMatrix({
      ...COV,
      labels: ["a", "b", "c"],
      covariance: [
        [4, -1.2, 0],
        [-1.2, 1, 0],
        [0, 0, 0],
      ],
    });
    expect(rho[0]![0]).toBe(1);
    expect(rho[0]![1]).toBeCloseTo(-0.6);
    expect(Number.isNaN(rho[2]![2])).toBe(true);
  });
});

describe("exportUncertainty", () => {
  const base: AnyExport = { per_feature_residuals: { target: [] } };
  it("reads the top level, then the laserline estimate", () => {
    const top = { status: "available", ...COV } as const;
    expect(exportUncertainty({ ...base, uncertainty: top })).toBe(top);
    expect(exportUncertainty({ ...base, estimate: { uncertainty: top } })).toBe(top);
    expect(exportUncertainty(base)).toBeNull();
  });
});

describe("unavailableText", () => {
  it("names the parameters the data does not determine", () => {
    expect(
      unavailableText({ kind: "rank_deficient", unobservable: ["cx", "tilt_y"] }),
    ).toBe("The data does not determine: cx, tilt_y.");
  });
});
