/** The editor's curated layout of a configuration schema: the fields that
 * matter most first, in groups, and every other field behind them, so the
 * curated view never hides a real option.
 *
 * It reads the schema and the config's own structure, never a topology's
 * name: a camera model is an object with a `model` (and a list of
 * `camera_overrides`), a robust loss is a field named `…_loss`, the solver
 * is `solver`, a stage's settings sit under `…_init`, and the switches that
 * free or fix an estimate are `refine_…` and `fix…`. Seven topologies, one
 * rule; a field no rule names lands in "All other fields". */
import {
  fieldsAt,
  getAtPath,
  setAtPath,
  shapeOf,
  type FieldUi,
  type JsonSchema,
  type UiGroup,
  type UiSchema,
} from "@vitavision/forms";
import { pathLabel } from "../../project/configEdit";
import { KIND_LABELS, LOSS_LABELS } from "./cameraParams";

interface Claims {
  camera: string[];
  fix: string[];
  loss: string[];
  solver: string[];
  init: string[];
  other: string[];
}

const join = (path: string, key: string) => (path === "" ? key : `${path}.${key}`);

function capitalised(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

/** The curated `UiSchema` of `schema` for the config `value`. */
export function curateConfig(schema: JsonSchema, value: unknown): UiSchema {
  const claims: Claims = {
    camera: [],
    fix: [],
    loss: [],
    solver: [],
    init: [],
    other: [],
  };
  const fields: Record<string, FieldUi> = {};

  const walk = (path: string, inInit: boolean, depth: number) => {
    if (depth > 6) return;
    for (const field of fieldsAt(schema, path, schema, value)) {
      const key = field.key;
      const p = join(path, key);
      const shape = shapeOf(field.schema, schema);
      const isObject = shape.kind === "object";
      if (key === "camera" && isObject && shape.fields.some((f) => f.key === "model")) {
        // The model's kinds belong with the camera; its fix lists with the
        // other switches that free or fix an estimate.
        claims.camera.push(`${p}.model`);
        fields[`${p}.model`] = {
          label: path === "" ? "Camera model" : `${pathLabel(path, 1)} camera model`,
        };
        cameraModelUi(p, fields);
        if (shape.fields.some((f) => f.key === "fix")) {
          claims.fix.push(`${p}.fix`);
          fields[`${p}.fix`] = { label: "Fixed parameters" };
        }
      } else if (key === "camera_overrides") {
        claims.camera.push(p);
        fields[p] = { label: "Per-camera overrides" };
        cameraModelUi(`${p}.*`, fields);
      } else if (key.endsWith("_loss")) {
        claims.loss.push(p);
        fields[p] = { label: capitalised(pathLabel(p)), enumLabels: LOSS_LABELS };
      } else if (key === "solver") {
        claims.solver.push(p);
        fields[p] = { label: capitalised(pathLabel(p)) };
      } else if (key === "manual_init") {
        claims.init.push(p);
        fields[p] = { label: capitalised(pathLabel(p)) };
      } else if (isObject) {
        walk(p, inInit || key.endsWith("_init"), depth + 1);
      } else {
        const group =
          key === "iterations" || key === "max_iters"
            ? claims.solver
            : /^(refine_|fix)/.test(key)
              ? claims.fix
              : inInit
                ? claims.init
                : claims.other;
        group.push(p);
        fields[p] = { label: capitalised(pathLabel(p)) };
      }
    }
  };
  walk("", false, 0);

  const groups: UiGroup[] = [
    { id: "camera", title: "Camera", fields: claims.camera },
    { id: "fix", title: "Fix and free", fields: claims.fix },
    { id: "loss", title: "Robust loss", fields: claims.loss },
    { id: "solver", title: "Solver and iterations", fields: claims.solver },
    { id: "init", title: "Initialization", fields: claims.init },
    {
      id: "other",
      title: "All other fields",
      fields: claims.other,
      collapsible: true,
      defaultOpen: false,
    },
  ].filter((g) => g.fields.length > 0);
  return { groups, fields };
}

/** The words for the kinds in a camera model's slots, at `model`'s path
 * (`camera`, or `camera_overrides.*`). */
function cameraModelUi(base: string, fields: Record<string, FieldUi>): void {
  for (const slot of ["projection", "distortion", "sensor"]) {
    fields[`${base}.model.${slot}`] = {
      label: capitalised(slot),
      enumLabels: KIND_LABELS,
    };
  }
}

/** The scale a robust loss starts at when its kind is switched to one that
 * has a scale: the schema seeds a required number with 0, which no loss
 * accepts. 1 is the library's own default for Huber and Cauchy (pixels). */
export const DEFAULT_LOSS_SCALE = 1;

/** `next` with the scale of every loss in `lossPaths` whose kind the edit
 * just switched to a scaled one set to `DEFAULT_LOSS_SCALE` where it is
 * not a positive number. Returns `next` itself when nothing changed. */
export function seedLossScales(
  prev: unknown,
  next: unknown,
  lossPaths: readonly string[],
): unknown {
  let result = next;
  for (const path of lossPaths) {
    const before = getAtPath(prev, `${path}.kind`);
    const kind = getAtPath(result, `${path}.kind`);
    if (kind === before || typeof kind !== "string" || kind === "none") continue;
    const scale = getAtPath(result, `${path}.scale`);
    if (typeof scale !== "number" || !(scale > 0)) {
      result = setAtPath(result, `${path}.scale`, DEFAULT_LOSS_SCALE);
    }
  }
  return result;
}
