import { describe, expect, it } from "vitest";
import { RUN_STAGES, computeStageRows, formatElapsed } from "./runStages";

describe("computeStageRows", () => {
  it("leaves everything pending before the first progress message", () => {
    const rows = computeStageRows(null);
    expect(rows.map((r) => r.state)).toEqual(["pending", "pending", "pending"]);
    expect(rows.map((r) => r.id)).toEqual(["detect", "solve", "export"]);
  });

  it("marks earlier stages done and the active one active", () => {
    const rows = computeStageRows("solve");
    expect(rows.map((r) => r.state)).toEqual(["done", "active", "pending"]);
  });

  it("marks the final stage active with the earlier two done", () => {
    const rows = computeStageRows("export");
    expect(rows.map((r) => r.state)).toEqual(["done", "done", "active"]);
  });

  it("marks the active and later stages cancelled, keeping done ones done", () => {
    const rows = computeStageRows("solve", { cancelled: true });
    expect(rows.map((r) => r.state)).toEqual(["done", "cancelled", "cancelled"]);
  });

  it("cancelling before any stage started marks all cancelled", () => {
    const rows = computeStageRows(null, { cancelled: true });
    // activeIdx = -1, so no stage is < -1; every stage is > -1 → cancelled.
    expect(rows.map((r) => r.state)).toEqual(["cancelled", "cancelled", "cancelled"]);
  });

  it("marks every stage done on successful completion regardless of last message", () => {
    const rows = computeStageRows("solve", { completed: true });
    expect(rows.map((r) => r.state)).toEqual(["done", "done", "done"]);
  });

  it("keeps the canonical stage vocabulary in lockstep with the runner", () => {
    expect(RUN_STAGES.map((s) => s.id)).toEqual(["detect", "solve", "export"]);
  });
});

describe("formatElapsed", () => {
  it("renders sub-minute times with one decimal second", () => {
    expect(formatElapsed(0)).toBe("0.0s");
    expect(formatElapsed(412)).toBe("0.4s");
    expect(formatElapsed(12_840)).toBe("12.8s");
    expect(formatElapsed(59_900)).toBe("59.9s");
  });

  it("renders minute+second times with a zero-padded seconds field", () => {
    expect(formatElapsed(63_000)).toBe("1m 03s");
    expect(formatElapsed(125_000)).toBe("2m 05s");
  });
});
