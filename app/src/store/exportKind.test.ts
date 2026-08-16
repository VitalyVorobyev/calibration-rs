import { describe, expect, it } from "vitest";
import { detectExportKind, exportKindLabel, type ExportKind } from "./exportKind";
import { PLANAR_EXPORT_FIXTURE } from "../test/exportFixtures";

const WIRE_KINDS: Exclude<ExportKind, "unknown">[] = [
  "planar_intrinsics",
  "scheimpflug_intrinsics",
  "single_cam_handeye",
  "laserline_device",
  "rig_extrinsics",
  "rig_handeye",
  "rig_laserline_device",
  "rig_handeye_laserline",
];

const ALL_KINDS: ExportKind[] = [...WIRE_KINDS, "unknown"];

// detectExportKind now reads the `kind` discriminator every pipeline
// `*Export` serializes, validated against the known vocabulary — no more
// probing which required fields are present.
describe("detectExportKind", () => {
  it("returns the wire kind for every recognised tag", () => {
    for (const kind of WIRE_KINDS) {
      expect(detectExportKind({ kind })).toBe(kind);
    }
  });

  it("classifies the shared planar fixture by its kind tag", () => {
    expect(detectExportKind(PLANAR_EXPORT_FIXTURE)).toBe("planar_intrinsics");
  });

  it("ignores field presence and trusts the tag", () => {
    // A payload whose fields look like a rig export but whose tag says
    // planar is classified by the tag: the discriminator is authoritative.
    expect(
      detectExportKind({
        kind: "planar_intrinsics",
        cameras: [],
        handeye_mode: "EyeInHand",
      }),
    ).toBe("planar_intrinsics");
  });

  it("returns unknown for a missing or unrecognised tag", () => {
    expect(detectExportKind({})).toBe("unknown");
    expect(detectExportKind({ kind: "not_a_real_kind" })).toBe("unknown");
    // Fields but no tag (e.g. a hand-edited payload, or one predating the tag): unknown.
    expect(detectExportKind({ cameras: [], cam_se3_rig: [], rig_se3_target: [] })).toBe(
      "unknown",
    );
  });

  it("returns unknown for non-object or non-string-tag payloads", () => {
    expect(detectExportKind(null)).toBe("unknown");
    expect(detectExportKind("nope")).toBe("unknown");
    expect(detectExportKind({ kind: 42 })).toBe("unknown");
    expect(detectExportKind({ kind: null })).toBe("unknown");
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
