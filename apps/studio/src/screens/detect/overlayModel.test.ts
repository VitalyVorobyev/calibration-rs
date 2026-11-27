import { describe, expect, it } from "vitest";
import {
  charucoDetail,
  chessboardDetail,
  puzzleboardDetail,
  ringgridDetail,
} from "./__fixtures__/details";
import {
  acceptedDetection,
  describeDiff,
  diffDetections,
  featureLabel,
  featureRows,
  rejectedDetection,
  rejectedItems,
  sigmaOf,
} from "./overlayModel";
import { pointsInBox } from "./pickModel";

describe("a chessboard", () => {
  const detail = chessboardDetail();

  it("draws its corners by feature index, labelled i,j, with the lattice kind", () => {
    const det = acceptedDetection(detail, { axes: false });
    expect(det.kind).toBe("chessboard");
    expect(det.corners).toHaveLength(4);
    expect(det.corners?.[1]).toMatchObject({
      id: 1,
      x: 120,
      y: 80,
      i: 1,
      j: 0,
      label: "1,0",
    });
    expect(det.corners?.[0]?.angle).toBeUndefined();
    expect(det.corners?.[0]?.cov).toEqual([0.0016, 0.0003, 0.0009]);
  });

  it("adds the corners' edge directions when asked", () => {
    const det = acceptedDetection(detail, { axes: true });
    expect(det.corners?.[0]).toMatchObject({ angle: 0, angle2: 1.57 });
  });

  it("lists its rejected candidates with the reason in words", () => {
    const items = rejectedItems(detail);
    expect(items).toHaveLength(1);
    expect(items[0]).toMatchObject({
      id: "r0",
      group: "unlabelled",
      sentence: "Corner candidate not taken by any labelled grid",
    });
    const det = rejectedDetection(items);
    expect(det.corners).toEqual([{ id: "r0", x: 300, y: 40, label: "" }]);
  });

  it("reads one row per feature with its σ from the covariance", () => {
    const rows = featureRows(detail);
    expect(rows).toHaveLength(4);
    expect(rows[3]).toMatchObject({
      index: 3,
      grid: [1, 1],
      score: 0.9,
      world: [0.02, 0.02, 0],
    });
    expect(rows[0]?.sigmaPx).toBeCloseTo(0.0414, 3);
  });
});

describe("a ChArUco board", () => {
  const detail = charucoDetail();

  it("draws the marker quads with their ids, and the misread cells as rejected", () => {
    const det = acceptedDetection(detail, { axes: false });
    expect(det.kind).toBe("charuco");
    expect(det.markers).toEqual([
      { id: "m0", corners: [120, 100, 140, 100, 140, 120, 120, 120], label: "12" },
    ]);
    const items = rejectedItems(detail);
    expect(items).toHaveLength(1);
    expect(items[0]).toMatchObject({
      id: "x0",
      kind: "marker",
      group: "marker",
      sentence: "Marker decodes as 17, board expects 23",
      x: 170,
      y: 110,
    });
    expect(rejectedDetection(items).markers?.[0]).toMatchObject({
      id: "x0",
      label: "17 ≠ 23",
    });
  });

  it("labels corners by their board cell and carries the ids", () => {
    const rows = featureRows(detail);
    expect(rows[0]).toMatchObject({ id: 5, grid: [1, 1] });
    expect(featureLabel("charuco", rows[0]!)).toBe("1,1");
  });
});

describe("a PuzzleBoard", () => {
  const detail = puzzleboardDetail();

  it("draws its edge bits at the edge midpoints, a quarter of the edge across", () => {
    const det = acceptedDetection(detail, { axes: false });
    expect(det.kind).toBe("puzzleboard");
    expect(det.edgeBits).toEqual([
      { id: "e0", x: 110, y: 80, radius: 5, bit: 1, confidence: 0.8 },
    ]);
  });

  it("labels corners by master id and flags both ends of a mismatching edge", () => {
    const rows = featureRows(detail);
    expect(featureLabel("puzzleboard", rows[0]!)).toBe("4021");
    expect(rows.every((r) => r.masterMismatch)).toBe(true);
    expect(featureRows(chessboardDetail()).some((r) => r.masterMismatch)).toBe(false);
  });
});

describe("a ring grid", () => {
  const detail = ringgridDetail();

  it("draws outer and inner ellipses at the centre, labelled with the code", () => {
    const det = acceptedDetection(detail, { axes: false });
    expect(det.kind).toBe("ringgrid");
    expect(det.rings).toHaveLength(2);
    expect(det.rings?.[0]).toMatchObject({
      id: 0,
      x: 100,
      y: 80,
      label: "17",
      outer: { rx: 9, ry: 8, angle: 0.2 },
      inner: { rx: 5, ry: 4, angle: 0.2 },
    });
    // A plain ring has no code: its lattice cell labels it, and no inner edge.
    expect(det.rings?.[1]?.inner).toBeUndefined();
    expect(det.rings?.[1]?.label).toBe("1,0");
    expect(det.circles).toEqual([
      { id: "o0", x: 90, y: 70, polarity: "white", label: "origin" },
    ]);
  });

  it("reads the centre estimator, confidence and covariance of each ring", () => {
    const rows = featureRows(detail);
    expect(rows[0]).toMatchObject({ center: "photometric", confidence: 0.93 });
    expect(rows[0]?.sigmaPx).toBeCloseTo(0.02);
    expect(rows[1]).toMatchObject({ center: "outer_ellipse", sigmaPx: null });
  });

  it("explains each rejected ring", () => {
    expect(rejectedItems(detail).map((i) => [i.group, i.sentence])).toEqual([
      ["unanchored", "Ring has no board position (view not anchored)"],
      ["outside_board", "Labelled outside the declared board"],
    ]);
  });
});

describe("sigmaOf", () => {
  it("is the larger principal standard deviation", () => {
    expect(sigmaOf([0.04, 0, 0.01])).toBeCloseTo(0.2);
    expect(sigmaOf(null)).toBeNull();
    expect(sigmaOf([-1, 0, -1])).toBeNull();
  });
});

describe("the diff of a preview against the saved detection", () => {
  const saved = chessboardDetail();

  it("counts features added, removed and relabelled", () => {
    const preview = chessboardDetail();
    preview.features = [
      ...preview.features.slice(0, 3),
      { image_xy: [400, 300], world_xyz: [0.5, 0.5, 0] },
    ];
    if (preview.target.kind !== "chessboard") throw new Error("a chessboard");
    preview.target.corners = [
      ...preview.target.corners.slice(0, 3),
      { grid: [5, 5], score: 0.5 },
    ];
    const diff = diffDetections(saved, preview);
    expect(diff).toEqual({ added: 1, removed: 1, relabelled: 0 });
    expect(describeDiff(diff)).toBe("+1 feature, −1 feature against the saved settings");
  });

  it("says when nothing changed", () => {
    expect(describeDiff(diffDetections(saved, chessboardDetail()))).toBe(
      "Same features as the saved settings",
    );
  });

  it("notices a corner that kept its place and changed its label", () => {
    const preview = chessboardDetail();
    preview.features[0] = { ...preview.features[0]!, world_xyz: [0.5, 0.5, 0] };
    expect(diffDetections(saved, preview).relabelled).toBe(1);
    expect(describeDiff(diffDetections(saved, preview))).toContain("1 label changed");
  });
});

describe("boxing features", () => {
  const points = [
    { index: 0, x: 10, y: 10 },
    { index: 1, x: 30, y: 10 },
    { index: 2, x: 10, y: 30 },
  ];

  it("takes those inside the box, whichever corner is first", () => {
    expect(pointsInBox(points, { x0: 0, y0: 0, x1: 20, y1: 20 })).toEqual([0]);
    expect(pointsInBox(points, { x0: 40, y0: 40, x1: 0, y1: 0 })).toEqual([0, 1, 2]);
    expect(pointsInBox(points, { x0: 50, y0: 50, x1: 60, y1: 60 })).toEqual([]);
  });
});
