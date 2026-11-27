import { describe, expect, it } from "vitest";
import type { Exclusions } from "../../api/generated/types";
import {
  countExclusions,
  describeChange,
  describeCounts,
  isFeatureExcluded,
  isImageOut,
  isViewExcluded,
  toggleFeatures,
  toggleImage,
  toggleView,
  unmatchedPoints,
  withNote,
  without,
} from "./exclusions";

const VIEW = { token: "0003", images: ["cam0/0003.png", "cam1/0003.png"] };
const NONE: Exclusions = {};

describe("views", () => {
  it("are left out by token and put back by the same gesture", () => {
    const out = toggleView(NONE, VIEW);
    expect(out.views).toEqual([{ view: { token: "0003" } }]);
    expect(isViewExcluded(out, VIEW)).toBe(true);
    expect(toggleView(out, VIEW).views).toEqual([]);
  });

  it("are found however the entry names them, and put back from either name", () => {
    const byImage: Exclusions = { views: [{ view: { image: "cam1/0003.png" } }] };
    expect(isViewExcluded(byImage, VIEW)).toBe(true);
    expect(isViewExcluded(byImage, { token: "0004", images: ["cam0/0004.png"] })).toBe(
      false,
    );
    expect(toggleView(byImage, VIEW).views).toEqual([]);
  });

  it("take every camera's image with them", () => {
    const out = toggleView(NONE, VIEW);
    expect(isImageOut(out, VIEW, "cam1", "cam1/0003.png")).toBe(true);
    expect(isImageOut(NONE, VIEW, "cam1", "cam1/0003.png")).toBe(false);
  });
});

describe("images", () => {
  it("are left out one camera at a time", () => {
    const one = toggleImage(NONE, "cam0", "cam0/0003.png");
    const two = toggleImage(one, "cam1", "cam1/0003.png");
    expect(two.images).toHaveLength(2);
    expect(isImageOut(two, VIEW, "cam0", "cam0/0003.png")).toBe(true);
    expect(toggleImage(two, "cam0", "cam0/0003.png").images).toEqual([
      { camera: "cam1", image: "cam1/0003.png" },
    ]);
  });
});

describe("features", () => {
  const a: [number, number, number] = [0.02, 0.04, 0];
  const b: [number, number, number] = [0.06, 0.04, 0];

  it("are left out by their target point in one entry per image", () => {
    const one = toggleFeatures(NONE, "cam0", "i.png", [a]);
    const two = toggleFeatures(one, "cam0", "i.png", [b]);
    expect(two.features).toEqual([{ camera: "cam0", image: "i.png", points: [a, b] }]);
    expect(isFeatureExcluded(two, "cam0", "i.png", a)).toBe(true);
    expect(isFeatureExcluded(two, "cam1", "i.png", a)).toBe(false);
  });

  it("match within a micrometre and are put back when all of them were out", () => {
    const one = toggleFeatures(NONE, "cam0", "i.png", [a, b]);
    expect(isFeatureExcluded(one, "cam0", "i.png", [0.02 + 5e-7, 0.04, 0])).toBe(true);
    expect(isFeatureExcluded(one, "cam0", "i.png", [0.02 + 5e-6, 0.04, 0])).toBe(false);
    expect(toggleFeatures(one, "cam0", "i.png", [a, b]).features).toEqual([]);
  });

  it("leave out the rest when only some of the picked were out", () => {
    const one = toggleFeatures(NONE, "cam0", "i.png", [a]);
    const both = toggleFeatures(one, "cam0", "i.png", [a, b]);
    expect(both.features?.[0]?.points).toHaveLength(2);
  });

  it("keep the entry's note and the other images' entries", () => {
    const start: Exclusions = {
      features: [
        { camera: "cam0", image: "i.png", points: [a], note: "glare" },
        { camera: "cam0", image: "j.png", points: [b] },
      ],
    };
    const out = toggleFeatures(start, "cam0", "i.png", [b]);
    expect(out.features?.find((e) => e.image === "i.png")).toEqual({
      camera: "cam0",
      image: "i.png",
      note: "glare",
      points: [a, b],
    });
    expect(out.features?.find((e) => e.image === "j.png")?.points).toEqual([b]);
  });

  it("do nothing without points", () => {
    expect(toggleFeatures(NONE, "cam0", "i.png", [])).toBe(NONE);
  });

  it("are listed as unmatched when no feature of the detection has the point", () => {
    const start = toggleFeatures(NONE, "cam0", "i.png", [a, b]);
    expect(unmatchedPoints(start, "cam0", "i.png", [a])).toEqual([b]);
    expect(unmatchedPoints(start, "cam1", "i.png", [])).toEqual([]);
  });
});

describe("the entries", () => {
  const start: Exclusions = {
    views: [{ view: { token: "1" } }],
    images: [{ camera: "cam0", image: "a.png" }],
    features: [{ camera: "cam0", image: "a.png", points: [[0, 0, 0]] }],
  };

  it("are counted and said", () => {
    expect(countExclusions(start)).toEqual({ views: 1, images: 1, features: 1 });
    expect(describeCounts(countExclusions(start))).toBe("1 view · 1 image · 1 feature");
    expect(describeCounts(countExclusions({}))).toBe("Nothing excluded");
  });

  it("take a note, and lose it when it is emptied", () => {
    const noted = withNote(start, "images", 0, " glare ");
    expect(noted.images?.[0]).toEqual({ camera: "cam0", image: "a.png", note: "glare" });
    expect(withNote(noted, "images", 0, "").images?.[0]).toEqual({
      camera: "cam0",
      image: "a.png",
    });
  });

  it("are restored one at a time", () => {
    expect(without(start, "views", 0).views).toEqual([]);
    expect(without(start, "views", 0).images).toHaveLength(1);
  });

  it("say what an operation changed", () => {
    expect(describeChange(NONE, toggleFeatures(NONE, "c", "i", [[0, 0, 0]]))).toBe(
      "Excluded 1 feature",
    );
    expect(describeChange(start, NONE)).toBe("Restored 1 view");
  });
});
