import { describe, expect, it } from "vitest";
import type {
  CameraParameter,
  CameraParameterSet,
  ParameterCovariance,
} from "../../api/generated/types";
import {
  groupParameters,
  modelSummary,
  otherRows,
  parametersJson,
  sigmaReason,
  viewOf,
} from "./parameterModel";

const p = (over: Partial<CameraParameter>): CameraParameter => ({
  at_bound: false,
  group: "intrinsics",
  label: "fx",
  name: "fx",
  sigma: { kind: "estimated", sigma: 0.21 },
  unit: "px",
  value: 1412.372,
  ...over,
});

describe("one parameter", () => {
  it("is value ± σ at the decimals of σ, with its unit and σ as a share of it", () => {
    const v = viewOf(p({}));
    expect(v.text).toBe("1412.37 ± 0.21");
    expect(v.unit).toBe("px");
    expect(v.relative).toBe("0.015 %");
    expect(v.reason).toBeNull();
  });

  it("is written with the reason when there is no σ", () => {
    const fixed = viewOf(
      p({
        name: "k3",
        group: "distortion",
        unit: "",
        value: 0,
        sigma: { kind: "unavailable", reason: { kind: "fixed" } },
      }),
    );
    expect(fixed.text).toBe("0 ·");
    expect(fixed.reason).toBe("held fixed, not estimated");
    expect(fixed.relative).toBeNull();
    const none = viewOf(
      p({
        sigma: {
          kind: "unavailable",
          reason: { kind: "no_covariance", reason: { kind: "not_converged" } },
        },
      }),
    );
    expect(none.reason).toMatch(/iteration budget/);
    expect(sigmaReason(p({}))).toBeNull();
    expect(
      sigmaReason(
        p({
          sigma: {
            kind: "unavailable",
            reason: { kind: "unsupported_model", message: "no such model" },
          },
        }),
      ),
    ).toBe("no such model");
    expect(
      sigmaReason(p({ sigma: { kind: "unavailable", reason: { kind: "not_listed" } } })),
    ).toMatch(/not among the estimated/);
  });

  it("writes a tilt in degrees, its σ too", () => {
    const v = viewOf(
      p({
        group: "sensor",
        name: "tilt_x",
        unit: "rad",
        value: 0.1,
        sigma: { kind: "estimated", sigma: 0.001 },
      }),
    );
    expect(v.unit).toBe("°");
    expect(v.text).toBe("5.730 ± 0.057");
  });
});

describe("groups", () => {
  it("run intrinsics, projection, distortion, sensor, then any other by name", () => {
    const groups = groupParameters([
      p({ group: "sensor", label: "tilt_x", name: "tilt_x" }),
      p({ group: "distortion", label: "k1", name: "k1" }),
      p({}),
      p({ group: "pose" as never, label: "r", name: "r" }),
    ]);
    expect(groups.map((g) => g.title)).toEqual([
      "Intrinsics",
      "Distortion",
      "Sensor",
      "pose",
    ]);
  });
});

describe("the model line", () => {
  it("names projection, distortion and a tilted sensor", () => {
    const set: CameraParameterSet = {
      camera: 0,
      distortion: "brown_conrady5",
      parameters: [],
      projection: "pinhole",
      sensor: "scheimpflug",
    };
    expect(modelSummary(set)).toBe("pinhole · Brown–Conrady (5) · tilted sensor");
    expect(
      modelSummary({
        ...set,
        sensor: "identity",
        projection: "equidistant",
        distortion: "kannala_brandt4",
      }),
    ).toBe("equidistant · Kannala–Brandt (4)");
  });
});

describe("copy as JSON", () => {
  it("holds the wire's values, σ or why there is none", () => {
    const set: CameraParameterSet = {
      camera: 1,
      distortion: "none",
      projection: "pinhole",
      parameters: [
        p({}),
        p({
          name: "k1",
          label: "cam1.k1",
          group: "distortion",
          unit: "",
          value: -0.2,
          sigma: { kind: "unavailable", reason: { kind: "fixed" } },
        }),
        p({
          name: "tilt_x",
          label: "tilt_x",
          group: "sensor",
          unit: "rad",
          value: 0.1,
          at_bound: true,
        }),
      ],
    };
    const json = JSON.parse(parametersJson(set)) as {
      camera: number;
      parameters: {
        name: string;
        value: number;
        unit: string;
        sigma: number | null;
        sigma_unavailable: string | null;
        at_bound: boolean;
      }[];
    };
    expect(json.camera).toBe(1);
    expect(json.parameters[0]).toMatchObject({
      name: "fx",
      value: 1412.372,
      sigma: 0.21,
      sigma_unavailable: null,
    });
    expect(json.parameters[1]).toMatchObject({
      sigma: null,
      sigma_unavailable: "held fixed, not estimated",
    });
    expect(json.parameters[2]).toMatchObject({ unit: "rad", value: 0.1, at_bound: true });
  });
});

describe("pose parameters", () => {
  const cov: ParameterCovariance = {
    labels: ["fx", "cam1.cam_se3_rig.tx", "cam2.cam_se3_rig.tx", "gripper_se3_rig.rz"],
    covariance: [
      [1, 0, 0, 0],
      [0, 1e-6, 0, 0],
      [0, 0, 1e-6, 0],
      [0, 0, 0, 1e-6],
    ],
    noise: [],
    num_residuals: 10,
    dof: 5,
  };
  it("are the shared ones no camera owns, of this camera and of none, σ only", () => {
    const rows = otherRows(cov, new Set(["fx"]), 1);
    expect(rows.map((r) => r.label)).toEqual([
      "cam1.cam_se3_rig.tx",
      "gripper_se3_rig.rz",
    ]);
    expect(rows.every((r) => r.value === null)).toBe(true);
    expect(rows[0]).toMatchObject({ unit: "mm" });
    expect(rows[0]!.sigma).toBeCloseTo(1);
    expect(rows[1]).toMatchObject({ unit: "°" });
  });
});
