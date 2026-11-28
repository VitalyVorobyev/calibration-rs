import { describe, expect, it } from "vitest";
import { runsOf, tabOf, withCompare } from "./compareParams";

const q = (s: string) => new URLSearchParams(s);

describe("the address of a comparison", () => {
  it("reads the runs in order, without repeats or blanks", () => {
    expect(runsOf(q("runs=run-002,run-001,,run-002"))).toEqual(["run-002", "run-001"]);
    expect(runsOf(q(""))).toEqual([]);
  });

  it("opens on the parameters unless the address names another tab", () => {
    expect(tabOf(q(""))).toBe("parameters");
    expect(tabOf(q("tab=maps"))).toBe("maps");
    expect(tabOf(q("tab=nonsense"))).toBe("parameters");
  });

  it("writes the runs and tab, keeping the project and run, and leaves out defaults", () => {
    const base = q("project=/p&run=run-003");
    const written = withCompare(base, ["a", "b"], "fit");
    expect(written.get("runs")).toBe("a,b");
    expect(written.get("tab")).toBe("fit");
    expect(written.get("project")).toBe("/p");
    expect(written.get("run")).toBe("run-003");
    const plain = withCompare(written, [], "parameters");
    expect(plain.has("runs")).toBe(false);
    expect(plain.has("tab")).toBe(false);
  });
});
