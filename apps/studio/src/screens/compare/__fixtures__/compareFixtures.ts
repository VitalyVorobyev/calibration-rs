/** A comparison reply for the Compare screen's tests: three runs of two
 * cameras. run-002 is an independent capture (every test is made, one
 * rejects), run-003 shares run-001's data (no test, with its reason, and a
 * different model for cam1). */
import type {
  CameraDifference,
  CameraEvidence,
  JointTest,
  ParameterDifference,
  ProjectionComparison,
  RunComparison,
  RunEvidence,
  Verdict,
} from "../../../api/generated/types";

export const DIR = "/data/bench";

const projection: ProjectionComparison = {
  status: "compared",
  compensation: "rotation",
  fit_nodes: 9,
  valid_nodes: 12,
  max_fit_px: 0.12,
  rms_fit_px: 0.05,
  max_valid_px: 0.3,
  rms_valid_px: 0.11,
  rotation_deg: 0.004,
};

const camera = (id: string, rms: number, views = 8): CameraEvidence => ({
  id,
  features: 400,
  views,
  unprojected: 0,
  image_size: [96, 72],
  max_projection_sigma_px: 0.2,
  reprojection: {
    count: 400,
    mean: rms * 0.9,
    rms,
    median: rms * 0.85,
    p95: rms * 2,
    max: rms * 3,
  },
});

const evidence = (name: string, verdict: Verdict, rms: number): RunEvidence => ({
  name,
  kind: "rig_extrinsics",
  verdict,
  reference_camera: "cam0",
  covariance: {
    status: "available",
    shared_parameters: 8,
    variance_factor: 1,
    view_scatter: 1.2,
  },
  cameras: [camera("cam0", rms), camera("cam1", rms * 1.05)],
});

const independent = (z: number, p: number) => ({
  kind: "independent" as const,
  z,
  p_value: p,
  sigma: 0.7,
});

const param = (
  label: string,
  reference: number,
  other: number,
  test: ParameterDifference["test"],
  sigma: number | null = 0.5,
): ParameterDifference => ({
  label,
  reference,
  reference_sigma: sigma,
  other,
  other_sigma: sigma,
  delta: other - reference,
  delta_over_reference_sigma: sigma ? (other - reference) / sigma : null,
  test,
});

const same = (
  parameters: ParameterDifference[],
  joint: JointTest,
): CameraDifference["parameters"] => ({ kind: "same_model", parameters, joint });

export function comparison(): RunComparison {
  return {
    runs: [
      evidence("run-001", "pass", 0.2),
      evidence("run-002", "pass", 0.22),
      evidence("run-003", "warn", 0.15),
    ],
    differences: [
      {
        run: 1,
        relation: { kind: "independent" },
        model_changes: [],
        unmatched_cameras: [],
        laser_planes: [],
        config_changes: null,
        cameras: [
          {
            id: "cam0",
            projection,
            parameters: same(
              [
                param("fx", 200, 200.35, independent(0.5, 0.62)),
                param("fy", 200, 202, independent(2.83, 0.005)),
                param(
                  "k3",
                  0,
                  0,
                  { kind: "not_available", reason: "not_estimated" },
                  null,
                ),
              ],
              { kind: "independent", chi2: 9.1, dof: 8, p_value: 0.33 },
            ),
          },
          {
            id: "cam1",
            projection,
            parameters: same([param("fx", 201, 200.9, independent(0.14, 0.89))], {
              kind: "independent",
              chi2: 21.4,
              dof: 8,
              p_value: 0.006,
            }),
          },
        ],
        rig: [
          {
            label: "cam1",
            rotation_deg: 0.05,
            translation_m: [0.0004, 0, 0],
            translation_norm_m: 0.0004,
            tangent: [0, 0, 0, 0, 0, 0],
            delta_over_reference_sigma: 1.1,
            test: { kind: "independent", chi2: 3.2, dof: 6, p_value: 0.78 },
          },
        ],
      },
      {
        run: 2,
        relation: { kind: "same_data" },
        model_changes: ["cam1"],
        unmatched_cameras: [],
        laser_planes: [],
        config_changes: [{ path: "solver.max_iters", reference: "50", other: "20" }],
        data: {
          shared_images: 8,
          shared_features: 3200,
          reference_only_images: 0,
          reference_only_features: 0,
          other_only_images: 0,
          other_only_features: 0,
        },
        cameras: [
          {
            id: "cam0",
            projection,
            parameters: same(
              [param("fx", 200, 200.1, { kind: "not_available", reason: "shared_data" })],
              { kind: "not_available", reason: "shared_data" },
            ),
          },
          {
            id: "cam1",
            projection: { status: "unavailable", reason: "too_few_nodes" },
            parameters: { kind: "different_models" },
          },
        ],
        rig: [],
      },
    ],
  };
}
