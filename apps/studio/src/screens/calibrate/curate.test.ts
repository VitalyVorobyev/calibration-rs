import type { JsonSchema } from "@vitavision/forms";
import { describe, expect, it } from "vitest";
import { curateConfig, seedLossScales } from "./curate";
import planar from "./__fixtures__/config_form_planar_intrinsics.json";
import singleCam from "./__fixtures__/config_form_single_cam_handeye.json";
import laser from "./__fixtures__/config_form_laserline_device.json";
import rigExtrinsics from "./__fixtures__/config_form_rig_extrinsics.json";
import rigHandeye from "./__fixtures__/config_form_rig_handeye.json";
import rigHandeyeLaser from "./__fixtures__/config_form_rig_handeye_laserline.json";
import rigLaser from "./__fixtures__/config_form_rig_laserline_device.json";

/** The seven workflows' config forms, as the library serves them. */
const FORMS: Record<string, { schema: JsonSchema; default: unknown }> = {
  planar_intrinsics: planar,
  single_cam_handeye: singleCam,
  laserline_device: laser,
  rig_extrinsics: rigExtrinsics,
  rig_handeye: rigHandeye,
  rig_handeye_laserline: rigHandeyeLaser,
  rig_laserline_device: rigLaser,
};

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

/** The paths of the default value's leaves (a list or empty table is one). */
function leaves(value: unknown, prefix = ""): string[] {
  if (!isRecord(value) || Object.keys(value).length === 0) return prefix ? [prefix] : [];
  return Object.entries(value).flatMap(([k, v]) =>
    leaves(v, prefix ? `${prefix}.${k}` : k),
  );
}

const related = (a: string, b: string) =>
  a === b || a.startsWith(`${b}.`) || b.startsWith(`${a}.`);

describe.each(Object.entries(FORMS))("curateConfig for %s", (_name, form) => {
  const ui = curateConfig(form.schema, form.default);
  const claimed = (ui.groups ?? []).flatMap((g) => g.fields);

  it("claims no field twice, nor a field and its parent", () => {
    for (let i = 0; i < claimed.length; i++) {
      for (let j = i + 1; j < claimed.length; j++) {
        expect(related(claimed[i]!, claimed[j]!), `${claimed[i]} and ${claimed[j]}`).toBe(
          false,
        );
      }
    }
  });

  it("covers every field of the default configuration", () => {
    for (const leaf of leaves(form.default)) {
      // `camera.model.projection` is under the claimed `camera.model`; an
      // empty table (`camera.fix`) has its names claimed one by one.
      const covered = claimed.some((c) => related(leaf, c));
      expect(covered, `${leaf} is in no group`).toBe(true);
    }
  });

  it("puts the other fields last and folded", () => {
    const groups = ui.groups ?? [];
    const other = groups.find((g) => g.id === "other");
    if (other) {
      expect(groups[groups.length - 1]).toBe(other);
      expect(other.collapsible).toBe(true);
      expect(other.defaultOpen).toBe(false);
    }
    expect(groups.every((g) => g.fields.length > 0)).toBe(true);
  });
});

describe("the curated groups", () => {
  it("find a camera, its fix lists, the losses and the solver, by structure", () => {
    const ui = curateConfig(planar.schema, planar.default);
    const group = (id: string) => ui.groups?.find((g) => g.id === id)?.fields ?? [];
    expect(group("camera")).toEqual(["camera.model"]);
    expect(group("fix")).toEqual(
      expect.arrayContaining(["camera.fix", "optimize.fix_poses"]),
    );
    expect(group("loss")).toEqual(["optimize.reprojection_loss"]);
    expect(group("solver")).toEqual(
      expect.arrayContaining(["solver", "intrinsics_init.iterations"]),
    );
  });

  it("reach the nested camera and every stage's loss of a rig with a laser", () => {
    const ui = curateConfig(rigHandeyeLaser.schema, rigHandeyeLaser.default);
    const group = (id: string) => ui.groups?.find((g) => g.id === id)?.fields ?? [];
    expect(group("camera")).toEqual(
      expect.arrayContaining(["handeye.camera.model", "handeye.camera_overrides"]),
    );
    expect(group("loss")).toEqual(
      expect.arrayContaining([
        "handeye.rig.reprojection_loss",
        "joint_optimize.laser_loss",
        "laserline_init.optimize.laser_loss",
      ]),
    );
  });

  it("word the kinds of a camera model, also in a list of overrides", () => {
    const ui = curateConfig(rigHandeye.schema, rigHandeye.default);
    expect(
      ui.fields?.["camera.model.distortion"]?.enumLabels?.["brown_conrady5"],
    ).toMatch(/Brown/);
    expect(
      ui.fields?.["camera_overrides.*.model.sensor"]?.enumLabels?.["scheimpflug"],
    ).toMatch(/Scheimpflug/);
  });
});

describe("seedLossScales", () => {
  const paths = ["optimize.reprojection_loss"];

  it("gives a loss switched to Huber a scale, not the schema's 0", () => {
    const prev = { optimize: { reprojection_loss: { kind: "none" } } };
    const next = { optimize: { reprojection_loss: { kind: "huber", scale: 0 } } };
    expect(seedLossScales(prev, next, paths)).toEqual({
      optimize: { reprojection_loss: { kind: "huber", scale: 1 } },
    });
  });

  it("leaves a scale the user typed, and a loss switched to none", () => {
    const prev = { optimize: { reprojection_loss: { kind: "huber", scale: 2 } } };
    const typed = { optimize: { reprojection_loss: { kind: "huber", scale: 3 } } };
    expect(seedLossScales(prev, typed, paths)).toBe(typed);
    const off = { optimize: { reprojection_loss: { kind: "none" } } };
    expect(seedLossScales(prev, off, paths)).toBe(off);
  });
});
