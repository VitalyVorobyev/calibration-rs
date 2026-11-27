import { describe, expect, it } from "vitest";
import { defaultFixed, fixChoices, fixSlots, toggledFix } from "./cameraParams";

const BC5 = { projection: "pinhole", distortion: "brown_conrady5", sensor: "identity" };
const TILTED = { ...BC5, sensor: "scheimpflug" };

describe("fixChoices", () => {
  it("names the parameters of each slot of a model", () => {
    expect(fixChoices("intrinsics", BC5)).toEqual(["fx", "fy", "cx", "cy"]);
    expect(fixChoices("distortion", BC5)).toEqual(["k1", "k2", "k3", "p1", "p2"]);
    expect(fixChoices("sensor", TILTED)).toEqual(["tilt_x", "tilt_y"]);
    expect(fixChoices("sensor", BC5)).toEqual([]);
    expect(fixChoices("projection", BC5)).toEqual([]);
  });

  it("does not guess a kind it does not know", () => {
    expect(fixChoices("distortion", { ...BC5, distortion: "future_kind" })).toBeNull();
    expect(fixSlots({ ...BC5, distortion: "future_kind" })).toBeNull();
  });
});

describe("fixSlots", () => {
  it("lists only the slots that have parameters", () => {
    expect(fixSlots(BC5)?.map((s) => s.slot)).toEqual(["intrinsics", "distortion"]);
    expect(fixSlots(TILTED)?.map((s) => s.slot)).toEqual([
      "intrinsics",
      "distortion",
      "sensor",
    ]);
  });
});

describe("defaultFixed", () => {
  it("is what the library holds fixed when the list is left out", () => {
    expect(defaultFixed("distortion", BC5)).toEqual(["k3"]);
    expect(defaultFixed("intrinsics", BC5)).toEqual([]);
    // Under a tilted sensor only the low-order radial terms stay free.
    expect(defaultFixed("distortion", TILTED)).toEqual(["k3", "p1", "p2"]);
    expect(defaultFixed("distortion", { ...BC5, distortion: "division1" })).toEqual([]);
  });
});

describe("toggledFix", () => {
  it("keeps the model's own order", () => {
    expect(toggledFix("distortion", BC5, ["k3"], "k1", true)).toEqual(["k1", "k3"]);
    expect(toggledFix("distortion", BC5, ["k1", "k3"], "k3", false)).toEqual(["k1"]);
  });
});
