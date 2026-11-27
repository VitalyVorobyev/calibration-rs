import { describe, expect, it } from "vitest";
import { INSPECT_TABS } from "../../shell/screens";
import { tabsFor } from "./tabs";

describe("Inspect's tabs", () => {
  it("are all offered while no run is shown", () => {
    expect(tabsFor(null)).toEqual([...INSPECT_TABS]);
  });

  it("leave out epipolar geometry and depth for a single camera", () => {
    const single = INSPECT_TABS.filter((t) => t !== "epipolar" && t !== "depth");
    expect(tabsFor(1)).toEqual(single);
    expect(tabsFor(0)).toEqual(single);
    expect(single).toContain("residuals");
  });

  it("are all there for a rig", () => {
    expect(tabsFor(2)).toEqual([...INSPECT_TABS]);
  });
});
