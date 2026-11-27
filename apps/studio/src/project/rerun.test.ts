import { describe, expect, it } from "vitest";
import type { Exclusions } from "../api/generated/types";
import { describeDifferences, exclusionDifferences } from "./rerun";

const image = (path: string) => ({ camera: "cam0", image: path });

describe("exclusionDifferences", () => {
  it("is empty for the same set in any order", () => {
    const a = { images: [image("1.png"), image("2.png")] } as unknown as Exclusions;
    const b = { images: [image("2.png"), image("1.png")] } as unknown as Exclusions;
    expect(exclusionDifferences(a, b)).toEqual([]);
    expect(exclusionDifferences(null, {})).toEqual([]);
  });

  it("counts what each side alone has, per kind", () => {
    const run = {
      images: [image("1.png")],
      views: [{ view: 3 }],
    } as unknown as Exclusions;
    const project = { images: [image("1.png"), image("2.png")] } as unknown as Exclusions;
    const diffs = exclusionDifferences(run, project);
    expect(diffs).toEqual([
      { kind: "images", onlyInRun: 0, onlyInProject: 1 },
      { kind: "views", onlyInRun: 1, onlyInProject: 0 },
    ]);
    expect(describeDifferences(diffs)).toBe(
      "images: 1 only the project has; views: 1 the run had",
    );
  });
});
