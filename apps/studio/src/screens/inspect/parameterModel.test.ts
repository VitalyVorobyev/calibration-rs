import { describe, expect, it } from "vitest";
import type {
  CameraParameter,
  CameraParameters,
  CameraParameterSet,
  ParameterCovariance,
  PoseParameter,
} from "../../api/generated/types";
import {
  coveredLabels,
  groupParameters,
  modelSummary,
  otherRows,
  parametersJson,
  planeView,
  poseView,
  sharedOf,
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

describe("pose and plane values", () => {
  const est = (sigma: number) => ({ kind: "estimated" as const, sigma });
  const pose: PoseParameter = {
    label: "cam1.cam_se3_rig",
    camera: 1,
    translation_m: [-0.12, 0.0001, 0],
    rotation_deg: [0.01, -0.5, 0],
    sigma: [est(0.01), est(0.02), est(0.03), est(0.0001), est(0.0002), est(0.0003)],
  };
  const reply: CameraParameters = {
    cameras: [],
    poses: [pose, { ...pose, label: "gripper_se3_rig", camera: null }],
    planes: [
      {
        label: "cam1.plane",
        camera: 1,
        normal: [0, 0.6, 0.8],
        distance_m: 0.5,
        sigma: [
          est(0.001),
          est(0.001),
          est(0.001),
          { kind: "unavailable", reason: { kind: "fixed" } },
        ],
      },
    ],
  };

  it("write a pose in millimetres and degrees, σ of its perturbation", () => {
    const v = poseView(pose);
    expect(v.translation).toBe("−120.000, 0.100, 0.000");
    expect(v.rotation).toBe("0.0100, −0.5000, 0.0000");
    expect(v.sigmaRotation).toBe("0.010, 0.020, 0.030");
    expect(v.sigmaTranslation).toBe("0.10, 0.20, 0.30");
    expect(v.reason).toBeNull();
    const none = poseView({
      ...pose,
      sigma: pose.sigma.map(() => ({
        kind: "unavailable" as const,
        reason: { kind: "not_listed" as const },
      })),
    });
    expect(none.sigmaRotation).toBe("–");
    expect(none.reason).toMatch(/not among the estimated/);
  });

  it("write a plane's normal and distance with their σ", () => {
    const v = planeView(reply.planes![0]!);
    expect(v.normal).toBe("0.0000 ± 0.0010, 0.6000 ± 0.0010, 0.8000 ± 0.0010");
    expect(v.distance).toBe("500 ·");
    expect(v.reason).toBe("held fixed, not estimated");
  });

  it("belong to their camera or to none, and cover their labels", () => {
    expect(sharedOf(reply, 0).poses.map((p) => p.label)).toEqual(["gripper_se3_rig"]);
    expect(sharedOf(reply, 1).poses).toHaveLength(2);
    expect(sharedOf(reply, 0).planes).toEqual([]);
    const covered = coveredLabels(reply);
    expect(covered.has("cam1.cam_se3_rig.tz")).toBe(true);
    expect(covered.has("gripper_se3_rig.rx")).toBe(true);
    expect(covered.has("cam1.plane.d")).toBe(true);
    expect(covered.has("base_se3_target.rx")).toBe(false);
  });
});
