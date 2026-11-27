import { describe, expect, it } from "vitest";
import type { DatasetSpec, StudioError } from "../../api/generated/types";
import {
  countProblems,
  datasetUi,
  derivedWarnings,
  formMessages,
  inferField,
  isManifestPath,
  patternMatch,
  placeMessages,
  toFormPath,
  usesPairing,
} from "./datasetModel";

const manifest = {
  version: 1,
  topology: "rig_extrinsics",
  target: { kind: "chessboard", rows: 4, cols: 5, square_size_m: 0.02 },
  cameras: [
    { id: "cam0", images: { kind: "glob", pattern: "images/cam0_*.png" } },
    { id: "cam1", images: { kind: "glob", pattern: "nothing/*.png" } },
  ],
} as unknown as DatasetSpec;

const err = (message: string, field?: string, suggestions?: string[]): StudioError => ({
  kind: "manifest",
  message,
  ...(field ? { field } : {}),
  ...(suggestions ? { suggestions } : {}),
});

describe("placing the library's messages", () => {
  it("turns the library's indexed paths into the form's, and knows the manifest's", () => {
    expect(toFormPath("cameras[1].images")).toBe("cameras.1.images");
    expect(toFormPath("pose_convention.transform")).toBe("pose_convention.transform");
    expect(isManifestPath("cameras.1.id")).toBe(true);
    expect(isManifestPath("device.id")).toBe(false);
    expect(isManifestPath("exclude.images.0")).toBe(false);
  });

  it("reads the field of a message that names none from its words", () => {
    const cams = manifest.cameras;
    expect(
      inferField(
        "manifest validation failed: invalid target: chessboard requires rows >= 2",
        cams,
      ),
    ).toBe("target");
    expect(
      inferField('manifest validation failed: duplicate camera id "cam0"', cams),
    ).toBe("cameras");
    expect(
      inferField(
        "manifest validation failed: topology RigExtrinsics expects 2..=18 camera(s), got 1",
        cams,
      ),
    ).toBe("cameras");
    expect(
      inferField(
        'manifest validation failed: camera "cam1" has degenerate ROI [0, 0, 0, 0]',
        cams,
      ),
    ).toBe("cameras.1.roi_xywh");
    expect(
      inferField('pose pairing regex "((" failed to compile: unclosed group', cams),
    ).toBe("pose_pairing.regex");
    expect(inferField('regex has no named capture group "nope"', cams)).toBe(
      "pose_pairing.group",
    );
    expect(inferField("something else entirely", cams)).toBeNull();
  });

  it("places a message on its field, and lists one about the project's own fields", () => {
    const placed = placeMessages(
      [
        err("Rig topologies need to know how images align.", "pose_pairing", [
          "by_index",
        ]),
        err("device.id: is empty", "device.id"),
        err("odd one"),
      ],
      manifest,
    );
    expect(placed[0]).toMatchObject({
      path: "pose_pairing",
      severity: "error",
      suggestions: ["by_index"],
    });
    expect(placed[1]!.path).toBeNull();
    expect(placed[2]!.path).toBeNull();
  });

  it("gives the form one message a field, an error never hidden by a warning", () => {
    const out = formMessages([
      {
        path: "cameras.1.images",
        severity: "warning",
        text: "No image matches.",
        suggestions: [],
      },
      { path: "cameras.1.images", severity: "error", text: "Bad glob.", suggestions: [] },
      { path: "target", severity: "warning", text: "Check.", suggestions: [] },
      { path: null, severity: "error", text: "Listed apart.", suggestions: [] },
    ]);
    expect(out["cameras.1.images"]).toMatchObject({ severity: "error" });
    expect(out["target"]).toEqual({ severity: "warning", text: "Check." });
    expect(Object.keys(out)).toEqual(["cameras.1.images", "target"]);
  });

  it("counts errors and warnings apart", () => {
    expect(
      countProblems([
        { path: null, severity: "error", text: "a", suggestions: [] },
        { path: null, severity: "warning", text: "b", suggestions: [] },
        { path: null, severity: "warning", text: "c", suggestions: [] },
      ]),
    ).toEqual({ errors: 1, warnings: 2 });
  });
});

describe("what a camera's pattern matched", () => {
  const reply = {
    cameras: ["cam0", "cam1"],
    views: [
      { token: "1", images: ["a/1.png", null] },
      { token: "2", images: ["a/2.png", null] },
      { token: "3", images: ["a/3.png", null] },
    ],
  };

  it("counts the files and names the first and last", () => {
    expect(patternMatch(reply, 0)).toEqual({
      files: 3,
      first: "a/1.png",
      last: "a/3.png",
    });
    expect(patternMatch(reply, 1)).toEqual({ files: 0, first: null, last: null });
  });

  it("warns on the pattern of a camera that matched nothing", () => {
    const warnings = derivedWarnings(reply, manifest);
    expect(warnings).toEqual([
      {
        path: "cameras.1.images",
        severity: "warning",
        text: "No image matches “nothing/*.png”.",
        suggestions: [],
      },
    ]);
    expect(derivedWarnings(null, manifest)).toEqual([]);
  });
});

describe("the form's layout", () => {
  it("hides the groups a workflow does not use and the bookkeeping fields", () => {
    const planar = datasetUi("planar_intrinsics", 1).fields!;
    expect(planar["robot_poses"]).toEqual({ hidden: true });
    expect(planar["laser"]).toEqual({ hidden: true });
    expect(planar["pose_pairing"]).toEqual({ hidden: true });
    expect(planar["_unresolved"]).toEqual({ hidden: true });
    const handeye = datasetUi("rig_handeye_laserline", 3).fields!;
    expect(handeye["robot_poses"]).toBeUndefined();
    expect(handeye["laser"]).toBeUndefined();
    expect(handeye["pose_pairing"]).toBeUndefined();
    expect(datasetUi("rig_extrinsics", 2).fields!["pose_pairing"]).toBeUndefined();
    expect(usesPairing("planar_intrinsics", 2)).toBe(true);
  });

  it("orders the groups: workflow, cameras, pairing, target, poses, laser, detector", () => {
    expect(datasetUi("rig_extrinsics", 2).groups!.map((g) => g.id)).toEqual([
      "workflow",
      "cameras",
      "pairing",
      "target",
      "poses",
      "laser",
      "detector",
      "notes",
    ]);
  });
});
