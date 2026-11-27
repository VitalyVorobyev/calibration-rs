/** The parameters of a camera model that a configuration can hold fixed,
 * and which of them the library holds fixed when the list is left out
 * (`null`). The names are the library's (`CameraFix`): a name the model
 * lacks is a configuration error the library reports, never one this
 * table hides.
 *
 * The camera model is the library's one composable model, shared by every
 * workflow, so this is a table of the model, not of a topology. */

/** A model as a configuration states it: the kind in each slot. */
export interface ModelKinds {
  projection?: string;
  distortion?: string;
  sensor?: string;
}

/** The slots of `CameraFix`. */
export const FIX_SLOTS = ["intrinsics", "projection", "distortion", "sensor"] as const;
export type FixSlot = (typeof FIX_SLOTS)[number];

const DISTORTION: Record<string, string[]> = {
  none: [],
  brown_conrady5: ["k1", "k2", "k3", "p1", "p2"],
  rational8: ["k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2"],
  thin_prism9: ["k1", "k2", "k3", "p1", "p2", "s1", "s2", "s3", "s4"],
  division1: ["lambda"],
  kannala_brandt4: ["k1", "k2", "k3", "k4"],
};

const SENSOR: Record<string, string[]> = {
  identity: [],
  scheimpflug: ["tilt_x", "tilt_y"],
};

/** The distortion terms that stay free under a tilted sensor: the
 * low-order radial ones (the rest trade off against the tilt). */
const LOW_ORDER = new Set(["k1", "k2", "lambda"]);

/** The names `slot` can fix under `model`; `null` when the model is one
 * this table does not know (the form then edits the list as text). */
export function fixChoices(slot: FixSlot, model: ModelKinds): string[] | null {
  switch (slot) {
    case "intrinsics":
      return ["fx", "fy", "cx", "cy"];
    case "projection":
      // Pinhole and equidistant have no parameters of their own.
      return model.projection === "pinhole" || model.projection === "equidistant"
        ? []
        : null;
    case "distortion":
      return DISTORTION[model.distortion ?? ""] ?? null;
    case "sensor":
      return SENSOR[model.sensor ?? ""] ?? null;
  }
}

/** What the library holds fixed in `slot` when the list is left out: `k3`
 * for Brown-Conrady, all but the low-order radial terms under a tilted
 * sensor, nothing else. `null` where this table cannot say. */
export function defaultFixed(slot: FixSlot, model: ModelKinds): string[] | null {
  if (slot !== "distortion") return [];
  const names = fixChoices("distortion", model);
  if (names === null) return null;
  if (model.sensor === "scheimpflug") return names.filter((n) => !LOW_ORDER.has(n));
  return model.distortion === "brown_conrady5" ? ["k3"] : [];
}

/** The chosen names of a fix list: a list of strings, else none. */
export function fixedNames(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((v): v is string => typeof v === "string")
    : [];
}

/** A fix list with `name` fixed or free, in the model's own order. */
export function toggledFix(
  slot: FixSlot,
  model: ModelKinds,
  current: readonly string[],
  name: string,
  fixed: boolean,
): string[] {
  const next = new Set(current);
  if (fixed) next.add(name);
  else next.delete(name);
  const order = fixChoices(slot, model) ?? [];
  return [...next].sort((a, b) => order.indexOf(a) - order.indexOf(b));
}

/** The names a model's distortion and sensor kinds read as. */
export const KIND_LABELS: Record<string, string> = {
  pinhole: "Pinhole",
  equidistant: "Equidistant (fisheye)",
  none: "None",
  brown_conrady5: "Brown–Conrady (k1 k2 k3 p1 p2)",
  rational8: "Rational (k1–k6 p1 p2)",
  thin_prism9: "Thin prism (Brown–Conrady + s1–s4)",
  division1: "Division (λ)",
  kannala_brandt4: "Kannala–Brandt (k1–k4)",
  identity: "Untilted",
  scheimpflug: "Scheimpflug (tilted)",
};

/** The kinds of each slot of the camera model, in the order a list shows
 * them. */
export const PROJECTION_KINDS = ["pinhole", "equidistant"] as const;
export const DISTORTION_KINDS = [
  "none",
  "brown_conrady5",
  "rational8",
  "thin_prism9",
  "division1",
  "kannala_brandt4",
] as const;
export const SENSOR_KINDS = ["identity", "scheimpflug"] as const;

/** The names of the robust-loss kinds. */
export const LOSS_LABELS: Record<string, string> = {
  none: "None (plain least squares)",
  huber: "Huber",
  cauchy: "Cauchy",
  arctan: "Arctangent",
};

export interface FixSlotChoices {
  slot: FixSlot;
  choices: string[];
}

/** The slots of `model` that have parameters to fix, with their names;
 * `null` when the table does not know the model (the form's own control
 * edits the lists then). */
export function fixSlots(model: ModelKinds): FixSlotChoices[] | null {
  const slots: FixSlotChoices[] = [];
  for (const slot of FIX_SLOTS) {
    const choices = fixChoices(slot, model);
    if (choices === null) return null;
    if (choices.length > 0) slots.push({ slot, choices });
  }
  return slots;
}
