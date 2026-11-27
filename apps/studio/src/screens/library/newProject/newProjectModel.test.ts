import type { JsonSchema } from "@vitavision/forms";
import { describe, expect, it } from "vitest";
import type { DatasetSpec } from "../../../api/generated/types";
import {
  answer,
  blockedAt,
  close,
  createSpec,
  dataFolder,
  defaultName,
  defaultProjectDir,
  isEditable,
  openFields,
  previewSpec,
  questionFor,
  slug,
  type CreateDraft,
} from "./newProjectModel";

const SCHEMA = {
  type: "object",
  properties: {
    pose_convention: { $ref: "#/$defs/PoseConvention" },
    robot_poses: { type: "object", properties: { path: { type: "string" } } },
    target: { type: "object", properties: { cols: { type: "integer" } } },
  },
  $defs: {
    PoseConvention: {
      type: "object",
      properties: {
        transform: { $ref: "#/$defs/Transform" },
        translation_units: { $ref: "#/$defs/Units" },
      },
    },
    Transform: { enum: ["t_base_tcp", "t_tcp_base"], type: "string" },
    Units: {
      oneOf: [
        { const: "m", type: "string" },
        { const: "mm", type: "string" },
      ],
    },
  },
} as unknown as JsonSchema;

const sniffed = {
  version: 1,
  topology: "single_cam_handeye",
  cameras: [{ id: "cam0", images: { kind: "glob", pattern: "*.png" } }],
  target: { kind: "chessboard", rows: 0, cols: 0, square_size_m: 0 },
  pose_convention: {
    transform: "t_base_tcp",
    translation_units: "m",
    rotation_format: "matrix4x4_row_major",
  },
  _unresolved: ["pose_convention.transform", "target"],
} as unknown as DatasetSpec;

describe("questions", () => {
  it("asks an enum field as a choice of its values, with what each means", () => {
    const q = questionFor("pose_convention.transform", sniffed, SCHEMA);
    expect(q.text).toBe("What does each robot pose row hold?");
    expect(q.control).toEqual({
      kind: "choice",
      options: [
        {
          value: "t_base_tcp",
          label: "t_base_tcp",
          note: "the tool's pose in the robot base frame",
        },
        {
          value: "t_tcp_base",
          label: "t_tcp_base",
          note: "the base's pose in the tool frame",
        },
      ],
    });
  });

  it("reads a oneOf of constants as choices too, and a string as text", () => {
    const units = questionFor("pose_convention.translation_units", sniffed, SCHEMA);
    expect(units.control).toMatchObject({ kind: "choice" });
    expect(
      units.control.kind === "choice" && units.control.options.map((o) => o.value),
    ).toEqual(["m", "mm"]);
    expect(questionFor("robot_poses.path", sniffed, SCHEMA).control).toEqual({
      kind: "text",
    });
  });

  it("sends the target to its own step and an unknown field to the full form", () => {
    expect(questionFor("target", sniffed, SCHEMA).control).toEqual({
      kind: "form",
      where: "the Target step",
    });
    const odd = questionFor("something.else", sniffed, SCHEMA);
    expect(odd.text).toBe("Set something.else");
    expect(odd.control.kind).toBe("form");
  });
});

describe("answering", () => {
  it("sets the value and closes the field, keeping the others open", () => {
    const next = answer(sniffed, "pose_convention.transform", "t_tcp_base");
    expect(next.pose_convention?.transform).toBe("t_tcp_base");
    expect(openFields(next)).toEqual(["target"]);
    expect(openFields(sniffed)).toEqual(["pose_convention.transform", "target"]);
  });

  it("drops _unresolved once nothing is open", () => {
    const next = close(
      answer(sniffed, "pose_convention.transform", "t_base_tcp"),
      "target",
    );
    expect("_unresolved" in next).toBe(false);
  });
});

describe("blockedAt", () => {
  const none = { product: "", id: "", label: "" };
  const folder = { kind: "folder" as const, folder: "/lib/set1" };

  it("holds each step until what it asks is answered", () => {
    expect(blockedAt("data", { source: null, manifest: null, device: none })).toMatch(
      /Pick a folder/,
    );
    expect(
      blockedAt("data", { source: folder, manifest: sniffed, device: none }),
    ).toBeNull();
    // The target is the next step's to settle, not the layout's.
    expect(
      blockedAt("layout", { source: folder, manifest: sniffed, device: none }),
    ).toMatch(/1 field is still open/);
    const decided = answer(sniffed, "pose_convention.transform", "t_base_tcp");
    expect(
      blockedAt("layout", { source: folder, manifest: decided, device: none }),
    ).toBeNull();
    expect(
      blockedAt("target", { source: folder, manifest: decided, device: none }),
    ).toMatch(/Describe the target/);
    expect(
      blockedAt("target", {
        source: folder,
        manifest: close(decided, "target"),
        device: none,
      }),
    ).toBeNull();
  });

  it("does not hold a referenced manifest it cannot edit, nor an empty device", () => {
    const referenced = { kind: "manifest" as const, path: "/lib/d.toml", embed: false };
    expect(
      blockedAt("layout", { source: referenced, manifest: sniffed, device: none }),
    ).toBeNull();
    expect(
      blockedAt("target", { source: referenced, manifest: sniffed, device: none }),
    ).toBeNull();
    expect(
      blockedAt("device", {
        source: referenced,
        manifest: sniffed,
        device: { ...none, product: "P" },
      }),
    ).toMatch(/needs an id/);
    expect(
      blockedAt("device", { source: referenced, manifest: sniffed, device: none }),
    ).toBeNull();
  });
});

describe("the project to create", () => {
  const draft = (over: Partial<CreateDraft>): CreateDraft => ({
    source: { kind: "folder", folder: "/lib/captures/set1" },
    manifest: sniffed,
    name: " Bench ",
    device: { product: "Stereo-X1", id: "SN1", label: "" },
    projectDir: "/lib/Stereo-X1/Bench",
    ...over,
  });

  it("writes the manifest inline with its data folder relative to the project", () => {
    const spec = createSpec(draft({}));
    expect(spec).toMatchObject({
      version: 2,
      name: "Bench",
      device: { id: "SN1", product: "Stereo-X1" },
      dataset: { root: "../../captures/set1" },
    });
    expect(spec.device).not.toHaveProperty("label");
  });

  it("references a manifest file, or embeds it with the folder it lies in", () => {
    const source = {
      kind: "manifest" as const,
      path: "/lib/captures/dataset.toml",
      embed: false,
    };
    const referenced = createSpec(
      draft({ source, device: { product: "", id: "", label: "" } }),
    );
    expect(referenced.dataset).toEqual({ manifest: "../../captures/dataset.toml" });
    expect(referenced.device).toBeUndefined();
    const embedded = createSpec(draft({ source: { ...source, embed: true } }));
    expect(embedded.dataset).toMatchObject({ root: "../../captures" });
  });

  it("previews against the data folder itself", () => {
    expect(previewSpec(sniffed).dataset).toMatchObject({ root: "." });
    expect(
      dataFolder({ kind: "manifest", path: "/a/b/dataset.toml", embed: false }),
    ).toBe("/a/b");
    expect(isEditable({ kind: "manifest", path: "/a/b.toml", embed: false })).toBe(false);
    expect(isEditable({ kind: "folder", folder: "/a" })).toBe(true);
  });
});

describe("names and device", () => {
  it("makes folder names and the default project folder", () => {
    expect(slug(" Line scanner / v2 ")).toBe("Line-scanner-v2");
    expect(defaultProjectDir("/lib", "Line scanner", "Bench 1")).toBe(
      "/lib/Line-scanner/Bench-1",
    );
    expect(defaultProjectDir("/lib", "", "Bench")).toBe("/lib/Bench");
    expect(defaultProjectDir("/lib", "", "///")).toBe("/lib/project");
    expect(defaultName({ kind: "folder", folder: "/lib/captures/set1/" })).toBe("set1");
  });
});
