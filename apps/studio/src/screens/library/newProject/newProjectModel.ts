/** The rules of the New project dialog, apart from its steps: what a
 * sniffed field the library could not decide is asked as (a question in
 * words, with the control its schema calls for), how an answer enters the
 * draft manifest, where the project file goes by default and the spec that
 * `create_project` is sent. */
import { fieldAt, getAtPath, setAtPath, type JsonSchema } from "@vitavision/forms";
import type { DatasetSpec, ProjectSpec } from "../../../api/generated/types";
import {
  deviceIdentity,
  deviceProblem,
  type DeviceDraft,
} from "../../../project/deviceDraft";
import { dirnamePath, joinPath, relativePath } from "../../../lib/paths";

export type NewStep = "data" | "layout" | "target" | "device" | "create";

export const NEW_STEPS: readonly { id: NewStep; label: string }[] = [
  { id: "data", label: "Data" },
  { id: "layout", label: "Layout" },
  { id: "target", label: "Target" },
  { id: "device", label: "Device" },
  { id: "create", label: "Create" },
];

/** Where the new project's data comes from. */
export type DataSource =
  /** A folder of images, its manifest drafted by the sniffer. */
  | { kind: "folder"; folder: string }
  /** An existing manifest file: referenced as it is, or copied into the
   * project file (`embed`) so it can be edited with it. */
  | { kind: "manifest"; path: string; embed: boolean };

/** The folder the manifest's relative paths resolve against. */
export function dataFolder(source: DataSource): string {
  return source.kind === "folder" ? source.folder : dirnamePath(source.path);
}

/** Whether the draft manifest can be edited here: a referenced manifest
 * file is never written. */
export function isEditable(source: DataSource): boolean {
  return source.kind === "folder" || source.embed;
}

/** The fields a draft manifest still leaves open (its `_unresolved`). */
export function openFields(spec: DatasetSpec): string[] {
  return spec._unresolved ?? [];
}

/** `spec` with `path` set to `value` and no longer open. */
export function answer(spec: DatasetSpec, path: string, value: unknown): DatasetSpec {
  const next = setAtPath(spec, path, value) as DatasetSpec;
  return close(next, path);
}

/** `spec` with `path` marked decided (a field the user filled in through a
 * form of its own, like the target). */
export function close(spec: DatasetSpec, path: string): DatasetSpec {
  const open = openFields(spec).filter((p) => p !== path);
  const { _unresolved: _dropped, ...rest } = spec;
  return open.length > 0 ? { ...rest, _unresolved: open } : rest;
}

/** The control an open field is asked with. */
export type QuestionControl =
  | { kind: "choice"; options: { value: string; label: string; note?: string }[] }
  | { kind: "text" }
  | { kind: "number" }
  /** Answered in the Target step or the full Dataset form. */
  | { kind: "form"; where: string };

export interface Question {
  path: string;
  /** The question, in words. */
  text: string;
  control: QuestionControl;
}

const QUESTIONS: Record<string, string> = {
  target: "What calibration target is in the images?",
  topology: "What are you calibrating?",
  pose_pairing: "How do the cameras' images pair into views?",
  robot_poses: "Which file holds the robot poses?",
  "robot_poses.path": "Which file holds the robot poses?",
  "robot_poses.format": "In which format are the robot poses stored?",
  "pose_convention.transform": "What does each robot pose row hold?",
  "pose_convention.translation_units": "In which units are the robot translations?",
  "pose_convention.rotation_format": "How is the robot rotation written?",
  upstream_calibration: "Which export holds the upstream calibration?",
};

/** What a choice's value says in words, where the code alone is cryptic. */
const OPTION_NOTES: Record<string, string> = {
  t_base_tcp: "the tool's pose in the robot base frame",
  t_tcp_base: "the base's pose in the tool frame",
  t_world_tcp: "the tool's pose in the world frame",
  t_tcp_world: "the world's pose in the tool frame",
  m: "metres",
  mm: "millimetres",
};

/** The question for the open field `path` of the draft `spec`, with the
 * control its schema in `schema` (the dataset form's) calls for. */
export function questionFor(
  path: string,
  spec: DatasetSpec,
  schema: JsonSchema | undefined,
): Question {
  const text = QUESTIONS[path] ?? `Set ${path}`;
  const info = schema ? fieldAt(schema, path, schema, spec) : undefined;
  const node = info?.schema as
    { enum?: unknown[]; oneOf?: unknown[]; $ref?: string; type?: string } | undefined;
  const options = node ? enumOptions(schema!, node) : null;
  if (options) {
    return {
      path,
      text,
      control: {
        kind: "choice",
        options: options.map((value) => {
          const note = OPTION_NOTES[value];
          return note ? { value, label: value, note } : { value, label: value };
        }),
      },
    };
  }
  if (path === "target") {
    return { path, text, control: { kind: "form", where: "the Target step" } };
  }
  if (node?.type === "number" || node?.type === "integer") {
    return { path, text, control: { kind: "number" } };
  }
  if (node?.type === "string" || path.endsWith("path")) {
    return { path, text, control: { kind: "text" } };
  }
  return { path, text, control: { kind: "form", where: "the full dataset form" } };
}

/** The values of a closed set of strings (an enum, a `oneOf` of consts),
 * following one `$ref`; `null` when the node is anything else. */
function enumOptions(root: JsonSchema, node: unknown): string[] | null {
  let n = node as Record<string, unknown>;
  const ref = n["$ref"];
  if (typeof ref === "string") {
    const key = ref.replace("#/$defs/", "");
    const defs = (root as unknown as { $defs?: Record<string, unknown> }).$defs;
    n = (defs?.[key] ?? {}) as Record<string, unknown>;
  }
  if (Array.isArray(n["enum"])) return (n["enum"] as unknown[]).map(String);
  const one = n["oneOf"];
  if (
    Array.isArray(one) &&
    one.every((v) => typeof (v as { const?: unknown }).const === "string")
  ) {
    return one.map((v) => (v as { const: string }).const);
  }
  return null;
}

/** The value now at `path`, for a control to show. */
export function valueAt(spec: DatasetSpec, path: string): unknown {
  return getAtPath(spec, path);
}

/** A folder name made of `text`: no separators or characters a file
 * system refuses, runs of white space one `-`. */
export function slug(text: string): string {
  return text
    .trim()
    .replace(/[\\/:*?"<>|]+/g, "-")
    .replace(/\s+/g, "-")
    .replace(/-{2,}/g, "-")
    .replace(/^[-.]+|[-.]+$/g, "");
}

/** Where the project file goes by default: `<library root>/<product>/<name>`
 * (`<library root>/<name>` without a product). */
export function defaultProjectDir(root: string, product: string, name: string): string {
  const parts = [slug(product), slug(name) || "project"].filter(Boolean);
  return joinPath(root, ...parts);
}

/** What the project is called when the user says nothing: the data
 * folder's (or manifest's folder's) name. */
export function defaultName(source: DataSource): string {
  const folder = dataFolder(source).replace(/[\\/]+$/, "");
  return folder.slice(folder.lastIndexOf("/") + 1);
}

/** The project folder the Create step shows and writes: the data folder
 * itself when asked, else the folder the user typed or chose, else the
 * default under the first library root (the data folder when there is no
 * root to put it under). */
export function resolveProjectDir(
  s: {
    source: DataSource;
    name: string | null;
    device: DeviceDraft;
    projectDir: string | null;
    inDataFolder: boolean;
  },
  libraryRoot: string | null,
): string {
  const folder = dataFolder(s.source);
  if (s.inDataFolder) return folder;
  if (s.projectDir !== null) return s.projectDir;
  if (libraryRoot === null) return folder;
  return defaultProjectDir(
    libraryRoot,
    s.device.product,
    s.name ?? defaultName(s.source),
  );
}

export interface CreateDraft {
  source: DataSource;
  manifest: DatasetSpec;
  name: string;
  device: DeviceDraft;
  /** The project folder. */
  projectDir: string;
}

/** The spec `create_project` is sent: the manifest inline (its data folder
 * against the project folder) or referenced, the device when given. */
export function createSpec(draft: CreateDraft): ProjectSpec {
  const spec: ProjectSpec = {
    version: 2,
    dataset:
      draft.source.kind === "manifest" && !draft.source.embed
        ? { manifest: relativePath(draft.projectDir, draft.source.path) }
        : {
            root: relativePath(draft.projectDir, dataFolder(draft.source)),
            manifest: draft.manifest,
          },
  };
  const name = draft.name.trim();
  if (name !== "") spec.name = name;
  const device = deviceIdentity(draft.device);
  if (device) spec.device = device;
  return spec;
}

/** The spec the layout preview asks `project_views` about: the manifest
 * inline, its data folder the one `project_views` is told (`.`). */
export function previewSpec(manifest: DatasetSpec): ProjectSpec {
  return { version: 2, dataset: { root: ".", manifest } };
}

/** Why `step` cannot be left forward yet; `null` when it can. */
export function blockedAt(
  step: NewStep,
  s: { source: DataSource | null; manifest: DatasetSpec | null; device: DeviceDraft },
): string | null {
  const open = s.manifest ? openFields(s.manifest) : [];
  const editable = s.source !== null && isEditable(s.source);
  switch (step) {
    case "data":
      return s.manifest ? null : "Pick a folder of images or a manifest first.";
    case "layout": {
      const left = open.filter((p) => p !== "target");
      return editable && left.length > 0
        ? `${left.length} ${left.length === 1 ? "field is" : "fields are"} still open.`
        : null;
    }
    case "target":
      return editable && open.includes("target") ? "Describe the target first." : null;
    case "device":
      return deviceProblem(s.device);
    case "create":
      return null;
  }
}
