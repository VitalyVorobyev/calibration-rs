import { describe, expect, it } from "vitest";
import { stageChecklist } from "./stages";

const states = (event: unknown, finished = false) =>
  stageChecklist(event, finished).map((s) => s.state);

describe("stageChecklist", () => {
  it("starts with detection active before any report", () => {
    expect(states(null)).toEqual(["active", "pending", "pending"]);
  });

  it("counts the images while detecting", () => {
    const rows = stageChecklist({ kind: "detection", done: 41, total: 120 }, false);
    expect(rows[0]).toMatchObject({ state: "active", detail: "41/120" });
    expect(rows[1]?.state).toBe("pending");
  });

  it("finishes detection when the solve reports, with its iteration", () => {
    const rows = stageChecklist(
      { kind: "iteration", iter: 12, max_iters: 50, cost: 0.32 },
      false,
    );
    expect(rows.map((r) => r.state)).toEqual(["done", "active", "pending"]);
    expect(rows[1]?.detail).toBe("iter 12/50 · cost 3.2e−1");
  });

  it("names a step", () => {
    expect(stageChecklist({ kind: "step", name: "rig init" }, false)[1]?.detail).toBe(
      "rig init",
    );
  });

  it("is all done when the job is", () => {
    expect(states({ kind: "iteration", iter: 3, max_iters: 3, cost: 1 }, true)).toEqual([
      "done",
      "done",
      "done",
    ]);
  });
});
