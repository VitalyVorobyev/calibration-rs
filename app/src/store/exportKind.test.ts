import { describe, expect, it } from "vitest";
import { detectExportKind, exportKindLabel, type ExportKind } from "./exportKind";

const ALL_KINDS: ExportKind[] = [
  "planar_intrinsics",
  "scheimpflug_intrinsics",
  "single_cam_handeye",
  "laserline_device",
  "rig_extrinsics",
  "rig_handeye",
  "rig_handeye_laserline",
  "rig_laserline_device",
  "unknown",
];

// detectExportKind classifies a loaded export by its distinguishing required
// fields — the actual shapes emitted by the pipeline (see the generated
// `*Export` interfaces), not the speculative top-level `camera` /
// `laser_planes_cam` probes of the retired exportShape.ts.
describe("detectExportKind", () => {
  it("classifies single-camera intrinsics by params.camera.sensor", () => {
    // Both planar and Scheimpflug exports are top-level identical; the sensor
    // model tag inside params.camera is the only discriminator.
    expect(
      detectExportKind({
        params: { camera: { sensor: { type: "identity" } } },
        report: { final_cost: 0 },
      }),
    ).toBe("planar_intrinsics");
    expect(
      detectExportKind({
        params: { camera: { sensor: { type: "scheimpflug" } } },
        report: { final_cost: 0 },
      }),
    ).toBe("scheimpflug_intrinsics");
    // Missing/omitted sensor tag falls back to planar (the common case).
    expect(detectExportKind({ params: { camera: {} } })).toBe("planar_intrinsics");
  });

  it("classifies single-camera hand-eye by camera + handeye_mode", () => {
    expect(detectExportKind({ camera: {}, handeye_mode: "EyeInHand" })).toBe(
      "single_cam_handeye",
    );
  });

  it("classifies a laserline device by estimate + stats", () => {
    expect(detectExportKind({ estimate: {}, stats: {} })).toBe("laserline_device");
  });

  it("classifies rig exports by the cameras array + optional hand-eye", () => {
    expect(detectExportKind({ cameras: [], cam_se3_rig: [], rig_se3_target: [] })).toBe(
      "rig_extrinsics",
    );
    expect(detectExportKind({ cameras: [], handeye_mode: "EyeToHand" })).toBe(
      "rig_handeye",
    );
  });

  it("classifies laser-rig exports by laser_planes_rig + optional hand-eye", () => {
    expect(
      detectExportKind({
        laser_planes_rig: [],
        laser_planes_cam: [],
        per_camera_stats: [],
      }),
    ).toBe("rig_laserline_device");
    expect(
      detectExportKind({ laser_planes_rig: [], handeye_mode: "EyeInHand", cameras: [] }),
    ).toBe("rig_handeye_laserline");
  });

  it("returns unknown for unrecognised or empty shapes", () => {
    expect(detectExportKind({})).toBe("unknown");
    expect(detectExportKind({ per_feature_residuals: {} })).toBe("unknown");
    expect(detectExportKind(null)).toBe("unknown");
    expect(detectExportKind("nope")).toBe("unknown");
  });

  // Probe order runs most-specific first: a laser-rig hand-eye export carries
  // `cameras` too, but the laser + hand-eye combination must win over the
  // plain rig / single-cam classifications.
  it("prefers the most specific kind when fields overlap", () => {
    expect(
      detectExportKind({
        laser_planes_rig: [],
        handeye_mode: "EyeInHand",
        cameras: [],
        camera: {},
      }),
    ).toBe("rig_handeye_laserline");
    expect(detectExportKind({ cameras: [], handeye_mode: "EyeToHand", camera: {} })).toBe(
      "rig_handeye",
    );
  });
});

describe("exportKindLabel", () => {
  it("labels every kind non-empty and distinctly", () => {
    const labels = ALL_KINDS.map(exportKindLabel);
    expect(new Set(labels).size).toBe(labels.length);
    for (const l of labels) expect(l.length).toBeGreaterThan(0);
  });

  it("labels unknown explicitly", () => {
    expect(exportKindLabel("unknown")).toBe("Unknown export");
  });
});
