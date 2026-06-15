import { describe, expect, it } from "vitest";
import { mergeConfig } from "./presets";

describe("mergeConfig", () => {
  it("returns the patch when the base is not a mergeable object", () => {
    expect(mergeConfig(5, { a: 1 })).toEqual({ a: 1 });
    expect(mergeConfig("x", { a: 1 })).toEqual({ a: 1 });
    expect(mergeConfig(null, { a: 1 })).toEqual({ a: 1 });
    // Arrays are not deep-merge targets — the patch replaces them.
    expect(mergeConfig([1, 2], { a: 1 })).toEqual({ a: 1 });
  });

  it("adds new keys and overrides scalar keys", () => {
    expect(mergeConfig({ a: 1 }, { b: 2 })).toEqual({ a: 1, b: 2 });
    expect(mergeConfig({ a: 1 }, { a: 2 })).toEqual({ a: 2 });
  });

  it("deep-merges nested objects", () => {
    expect(mergeConfig({ a: { x: 1, y: 2 } }, { a: { y: 3, z: 4 } })).toEqual({
      a: { x: 1, y: 3, z: 4 },
    });
  });

  it("replaces arrays instead of merging them", () => {
    expect(mergeConfig({ a: [1, 2, 3] }, { a: [9] })).toEqual({ a: [9] });
  });

  it("replaces when base and patch types disagree at a key", () => {
    // object base, scalar patch → scalar wins
    expect(mergeConfig({ a: { x: 1 } }, { a: 5 })).toEqual({ a: 5 });
    // scalar base, object patch → object wins
    expect(mergeConfig({ a: 1 }, { a: { x: 1 } })).toEqual({ a: { x: 1 } });
  });

  it("treats a null patch value as an explicit override (not a deep merge)", () => {
    // Mirrors RTV3D_JOINT_LASER_MANIFEST_OVERRIDES.upstream_calibration: null.
    expect(mergeConfig({ upstream: { id: 1 } }, { upstream: null })).toEqual({
      upstream: null,
    });
  });

  it("does not mutate the base object", () => {
    const base = { a: { x: 1 }, keep: 7 };
    const frozen = structuredClone(base);
    const out = mergeConfig(base, { a: { y: 2 } });
    expect(base).toEqual(frozen); // base untouched
    expect(out).toEqual({ a: { x: 1, y: 2 }, keep: 7 });
    expect(out).not.toBe(base);
  });

  it("applies a realistic two-level config override", () => {
    const defaults = {
      sensor: { kind: "Pinhole", init_tilt_x: 0 },
      solver: { max_iters: 100 },
    };
    const overrides = {
      sensor: { kind: "Scheimpflug" },
      handeye_init: { handeye_mode: "EyeToHand" },
    };
    expect(mergeConfig(defaults, overrides)).toEqual({
      sensor: { kind: "Scheimpflug", init_tilt_x: 0 },
      solver: { max_iters: 100 },
      handeye_init: { handeye_mode: "EyeToHand" },
    });
  });
});
