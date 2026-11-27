import { describe, expect, it } from "vitest";
import type { QualityCheck } from "../../api/generated/types";
import {
  CHECK_WORDS,
  checksFor,
  measurementText,
  measureText,
  statusTone,
  tally,
} from "./qualityModel";

const check = (over: Partial<QualityCheck>): QualityCheck => ({
  kind: "projection_sigma",
  status: "pass",
  message: "m",
  ...over,
});

describe("measurements", () => {
  it("are written with their unit and the limit they were held to", () => {
    expect(measurementText({ value: 0.3142, threshold: 1, unit: "pixel" })).toBe(
      "measured 0.314 px, limit 1 px",
    );
    expect(measurementText({ value: 0.167, threshold: 0.3, unit: "fraction" })).toBe(
      "measured 16.7 %, limit 30 %",
    );
    expect(measurementText({ value: 2.04, threshold: 3, unit: "sigma" })).toBe(
      "measured 2.04 σ̂, limit 3 σ̂",
    );
    expect(measureText(0.97, "correlation")).toBe("0.97");
    expect(measurementText(null)).toBeNull();
    expect(measurementText(undefined)).toBeNull();
  });
});

describe("the checks", () => {
  it("have words for every kind the report can hold", () => {
    for (const w of Object.values(CHECK_WORDS)) {
      expect(w.title.length).toBeGreaterThan(0);
      expect(w.about.endsWith(".")).toBe(true);
    }
    expect(Object.keys(CHECK_WORDS)).toHaveLength(9);
  });

  it("keep the whole result's and the chosen camera's", () => {
    const all = [
      check({ kind: "convergence" }),
      check({ camera: 0 }),
      check({ camera: 1, status: "fail" }),
    ];
    expect(checksFor(all, 1).map((c) => c.camera ?? null)).toEqual([null, 1]);
    expect(checksFor(all, null)).toHaveLength(3);
  });

  it("are tallied by status, and verdict colour is by status only", () => {
    const t = tally([
      check({}),
      check({ status: "warn" }),
      check({ status: "not_applicable" }),
      check({ status: "fail" }),
    ]);
    expect(t).toEqual({ pass: 1, warn: 1, fail: 1, not_applicable: 1 });
    expect(statusTone("pass")).toBe("normal");
    expect(statusTone("warn")).toBe("warning");
    expect(statusTone("fail")).toBe("defect");
    expect(statusTone("not_applicable")).toBe("neutral");
  });
});
