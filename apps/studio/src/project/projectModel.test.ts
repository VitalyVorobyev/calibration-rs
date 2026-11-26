import { describe, expect, it } from "vitest";
import { folderName, nextRunName } from "./projectModel";

describe("nextRunName", () => {
  it("starts at run-001", () => {
    expect(nextRunName([])).toBe("run-001");
  });

  it("goes one past the highest numbered run, ignoring other names", () => {
    expect(nextRunName(["run-001", "run-007", "baseline", "run-x"])).toBe("run-008");
  });

  it("does not reuse a gap", () => {
    expect(nextRunName(["run-003"])).toBe("run-004");
  });

  it("grows past three digits", () => {
    expect(nextRunName(["run-999"])).toBe("run-1000");
  });
});

describe("folderName", () => {
  it("takes the last segment of either separator, trailing ones ignored", () => {
    expect(folderName("/data/kuka_1")).toBe("kuka_1");
    expect(folderName("/data/kuka_1/")).toBe("kuka_1");
    expect(folderName("C:\\data\\rig")).toBe("rig");
    expect(folderName("plain")).toBe("plain");
  });
});
