import { describe, expect, it } from "vitest";
import { tabsFor } from "./tabs";

describe("Inspect's tabs", () => {
  it("are all offered while no run is shown", () => {
    expect(tabsFor(null)).toEqual(["scene", "frames", "epipolar", "depth"]);
  });

  it("leave out epipolar geometry and depth for a single camera", () => {
    expect(tabsFor(1)).toEqual(["scene", "frames"]);
    expect(tabsFor(0)).toEqual(["scene", "frames"]);
  });

  it("are all there for a rig", () => {
    expect(tabsFor(2)).toEqual(["scene", "frames", "epipolar", "depth"]);
  });
});
