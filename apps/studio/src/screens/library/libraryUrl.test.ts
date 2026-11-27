import { describe, expect, it } from "vitest";
import { fromParams, toParams } from "./libraryUrl";

describe("libraryUrl", () => {
  it("reads the filter and selection from the address", () => {
    const view = fromParams(
      new URLSearchParams("q=bench&v=warn,fail,bogus&w=rig_handeye&sel=device:X"),
    );
    expect(view.filter.query).toBe("bench");
    expect([...view.filter.verdicts]).toEqual(["warn", "fail"]);
    expect(view.filter.topology).toBe("rig_handeye");
    expect(view.selected).toBe("device:X");
  });

  it("round-trips and leaves the project and run parameters alone", () => {
    const params = new URLSearchParams("project=/p&run=run-001&q=old");
    const view = fromParams(new URLSearchParams("q=a b&v=none,pass&w=planar_intrinsics"));
    const next = toParams(params, { ...view, selected: "ungrouped" });
    expect(next.get("project")).toBe("/p");
    expect(next.get("run")).toBe("run-001");
    expect(next.get("q")).toBe("a b");
    expect(next.get("v")).toBe("pass,none");
    expect(fromParams(next)).toEqual({ ...view, selected: "ungrouped" });
  });

  it("drops the keys at their defaults", () => {
    const cleared = toParams(new URLSearchParams("q=x&v=pass&w=rig_handeye&sel=a"), {
      filter: { query: "", verdicts: new Set(), topology: null },
      selected: null,
    });
    expect([...cleared.keys()]).toEqual([]);
  });
});
