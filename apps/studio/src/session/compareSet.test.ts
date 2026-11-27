import { beforeEach, describe, expect, it } from "vitest";
import { pruned, toggled, useCompareSet } from "./compareSet";

describe("compare set rules", () => {
  it("toggles in the order of ticking", () => {
    expect(toggled([], "run-002")).toEqual(["run-002"]);
    expect(toggled(["run-002"], "run-001")).toEqual(["run-002", "run-001"]);
    expect(toggled(["run-002", "run-001"], "run-002")).toEqual(["run-001"]);
  });

  it("drops the runs that are gone, keeping the array when nothing is", () => {
    const ids = ["run-001", "run-002"];
    expect(pruned(ids, ["run-001", "run-002", "run-003"])).toBe(ids);
    expect(pruned(ids, ["run-002"])).toEqual(["run-002"]);
  });
});

describe("useCompareSet", () => {
  beforeEach(() => useCompareSet.setState({ dir: null, ids: [] }));

  it("belongs to one project", () => {
    const { toggle, follow } = useCompareSet.getState();
    toggle("/a", "run-001");
    toggle("/a", "run-002");
    expect(useCompareSet.getState().ids).toEqual(["run-001", "run-002"]);
    toggle("/b", "run-001");
    expect(useCompareSet.getState()).toMatchObject({ dir: "/b", ids: ["run-001"] });
    follow("/a", ["run-001"]);
    expect(useCompareSet.getState()).toMatchObject({ dir: "/a", ids: [] });
  });

  it("prunes a deleted run", () => {
    const { toggle, follow } = useCompareSet.getState();
    toggle("/a", "run-001");
    toggle("/a", "run-002");
    follow("/a", ["run-002"]);
    expect(useCompareSet.getState().ids).toEqual(["run-002"]);
  });
});
