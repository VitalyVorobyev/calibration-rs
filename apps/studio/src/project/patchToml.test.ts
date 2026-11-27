import { describe, expect, it } from "vitest";
import { patchToml } from "./patchToml";

describe("patchToml", () => {
  it("writes one dotted key per leaf, in key order", () => {
    expect(
      patchToml({
        solver: { max_iters: 20 },
        optimize: { reprojection_loss: { kind: "huber", scale: 1.5 } },
      }),
    ).toEqual([
      'optimize.reprojection_loss.kind = "huber"',
      "optimize.reprojection_loss.scale = 1.5",
      "solver.max_iters = 20",
    ]);
  });

  it("writes lists, lists of tables and empty tables", () => {
    expect(
      patchToml({
        camera: { fix: { distortion: ["k3", "p1"] } },
        camera_overrides: [{ camera: 1, model: { sensor: "scheimpflug" } }],
        extra: {},
      }),
    ).toEqual([
      'camera.fix.distortion = ["k3", "p1"]',
      'camera_overrides = [{ camera = 1, model = { sensor = "scheimpflug" } }]',
      "extra = {}",
    ]);
  });

  it("comments a field set back to none, which TOML cannot say", () => {
    expect(patchToml({ manual_init: null })).toEqual([
      "# manual_init = none (JSON only)",
    ]);
  });

  it("is empty for an empty patch", () => {
    expect(patchToml({})).toEqual([]);
  });
});
