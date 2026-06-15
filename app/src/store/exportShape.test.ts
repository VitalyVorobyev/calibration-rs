import { describe, expect, it } from "vitest";
import { exportKindLabel, inferExportKind } from "./exportShape";
import type { AnyExport, ExportKind } from "./types";

// inferExportKind probes a loose JSON shape and never reads
// per_feature_residuals, so tests build minimal records and cast. Using
// `unknown` keeps the probe fields (sensor, scheimpflug, laser_plane_cam)
// that live outside the AnyExport interface but are part of the shape
// contract.
const ex = (obj: Record<string, unknown>): AnyExport => obj as unknown as AnyExport;

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

describe("inferExportKind", () => {
  it("classifies each export shape", () => {
    expect(inferExportKind(ex({ camera: {} }))).toBe("planar_intrinsics");
    expect(inferExportKind(ex({ camera: {}, sensor: {} }))).toBe(
      "scheimpflug_intrinsics",
    );
    expect(inferExportKind(ex({ camera: {}, scheimpflug: {} }))).toBe(
      "scheimpflug_intrinsics",
    );
    expect(inferExportKind(ex({ handeye_mode: "EyeInHand" }))).toBe(
      "single_cam_handeye",
    );
    expect(inferExportKind(ex({ laser_planes_cam: [] }))).toBe("laserline_device");
    expect(inferExportKind(ex({ laser_plane_cam: {} }))).toBe("laserline_device");
    expect(inferExportKind(ex({ cameras: [] }))).toBe("rig_extrinsics");
    expect(inferExportKind(ex({ cameras: [], handeye_mode: "EyeToHand" }))).toBe(
      "rig_handeye",
    );
    expect(inferExportKind(ex({ laser_planes_rig: [] }))).toBe(
      "rig_laserline_device",
    );
    expect(
      inferExportKind(ex({ laser_planes_rig: [], handeye_mode: "EyeToHand" })),
    ).toBe("rig_handeye_laserline");
  });

  it("returns 'unknown' for an unrecognized shape", () => {
    expect(inferExportKind(ex({}))).toBe("unknown");
    expect(inferExportKind(ex({ per_feature_residuals: {} }))).toBe("unknown");
  });

  // The probe order is load-bearing: rig exports also expose legacy
  // top-level `camera`-prefixed fields, so rig shapes must be matched
  // before single-cam shapes (see the inferExportKind doc comment).
  describe("probe order is shape-precedence safe", () => {
    it("rig + laser + handeye + legacy camera → rig_handeye_laserline", () => {
      expect(
        inferExportKind(
          ex({
            laser_planes_rig: [],
            handeye_mode: "EyeToHand",
            cameras: [],
            camera: {},
            sensor: {},
          }),
        ),
      ).toBe("rig_handeye_laserline");
    });

    it("rig laser without handeye → rig_laserline_device, not laserline_device", () => {
      expect(
        inferExportKind(ex({ laser_planes_rig: [], laser_planes_cam: [], camera: {} })),
      ).toBe("rig_laserline_device");
    });

    it("rig hand-eye with legacy camera → rig_handeye, not single_cam_handeye", () => {
      expect(
        inferExportKind(ex({ cameras: [], handeye_mode: "EyeToHand", camera: {} })),
      ).toBe("rig_handeye");
    });

    it("rig extrinsics with legacy camera+sensor → rig_extrinsics, not scheimpflug", () => {
      expect(inferExportKind(ex({ cameras: [], camera: {}, sensor: {} }))).toBe(
        "rig_extrinsics",
      );
    });

    it("hand-eye single-cam outranks a bare camera shape", () => {
      expect(inferExportKind(ex({ handeye_mode: "EyeInHand", camera: {} }))).toBe(
        "single_cam_handeye",
      );
    });
  });

  // A scheimpflug discriminator (sensor/scheimpflug) without `camera` is
  // not enough — the planar/scheimpflug probes both require `camera`.
  it("requires `camera` for the intrinsics shapes", () => {
    expect(inferExportKind(ex({ sensor: {} }))).toBe("unknown");
    expect(inferExportKind(ex({ scheimpflug: {} }))).toBe("unknown");
  });
});

describe("exportKindLabel", () => {
  it("maps every kind to a non-empty, unique label", () => {
    const labels = ALL_KINDS.map(exportKindLabel);
    for (const label of labels) {
      expect(label.length).toBeGreaterThan(0);
    }
    expect(new Set(labels).size).toBe(ALL_KINDS.length);
  });

  it("labels the unknown kind", () => {
    expect(exportKindLabel("unknown")).toBe("Unknown export");
  });
});
