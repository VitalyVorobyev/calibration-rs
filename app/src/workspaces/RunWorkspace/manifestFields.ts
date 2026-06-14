/** Helpers for the B3d manifest UX: vendor-aware guidance for fields the
 * sniffer or runner could not resolve, plus small manifest mutators used by
 * the unresolved notice and the AskUser modal. */

/** Human- and vendor-aware guidance keyed by manifest field path. Shown
 * beside `_unresolved` entries and in the AskUser modal so the user knows
 * what to fill in and why the app refused to guess (ADR 0019). */
export const FIELD_HINTS: Record<string, string> = {
  target:
    "Set the calibration target — kind (chessboard / charuco / puzzleboard / ringgrid) and geometry (rows/cols + square size in metres). Read these off the printed target or its spec sheet.",
  topology:
    "Confirm the calibration topology. Images alone can't distinguish a multi-camera rig from independent single-camera calibrations, nor a pinhole sensor from a Scheimpflug-tilted one.",
  pose_pairing:
    "How per-camera images align into views: by_index (same count and order everywhere) or shared_filename_token (a regex extracts a shared view token from filenames).",
  pose_convention:
    "Set the robot-pose frame convention — transform direction, rotation format, and translation units.",
  "pose_convention.transform":
    "Which transform each robot-pose row encodes. KUKA / UR export t_base_tcp (gripper-in-base); some ABB export t_tcp_base. Check your robot's export convention.",
  "pose_convention.translation_units":
    "Translation units in the pose file: m (SI, preferred) or mm (some ABB / Fanuc exports).",
  "robot_poses.columns":
    "Map your pose file's columns to tx / ty / tz + rotation (and an optional pose_id for token pairing).",
  "robot_poses.columns.pose_id":
    "shared_filename_token pairing needs a pose_id column so each pose row can match its view token.",
};

/** Guidance for a field path, falling back to its dotted-prefix parent. */
export function hintFor(path: string): string | undefined {
  if (FIELD_HINTS[path]) return FIELD_HINTS[path];
  // Fall back to a parent prefix (e.g. "pose_convention.foo" → "pose_convention").
  const prefix = path.split(".").slice(0, -1).join(".");
  return prefix ? FIELD_HINTS[prefix] : undefined;
}

/** Read the `_unresolved` string array off a manifest value. */
export function unresolvedPaths(manifest: unknown): string[] {
  const m = manifest as Record<string, unknown> | null;
  const u = m?._unresolved;
  return Array.isArray(u) ? u.filter((x): x is string => typeof x === "string") : [];
}

/** Return a copy of `manifest` with `path` removed from `_unresolved`. The
 * key is dropped entirely once the list empties so the serialized manifest
 * matches a hand-authored one (which omits `_unresolved`). */
export function clearUnresolved(manifest: unknown, path: string): unknown {
  const m = { ...(manifest as Record<string, unknown>) };
  const next = unresolvedPaths(m).filter((p) => p !== path);
  if (next.length === 0) {
    delete m._unresolved;
  } else {
    m._unresolved = next;
  }
  return m;
}

/** Set a value at a dotted `path` in a JSON-compatible object, returning a
 * deep copy. Intermediate objects are created as needed. */
export function setAtPath(obj: unknown, path: string, value: unknown): unknown {
  const root = JSON.parse(JSON.stringify(obj ?? {})) as Record<string, unknown>;
  const keys = path.split(".");
  let cur = root;
  for (let i = 0; i < keys.length - 1; i++) {
    const k = keys[i];
    if (typeof cur[k] !== "object" || cur[k] === null) cur[k] = {};
    cur = cur[k] as Record<string, unknown>;
  }
  cur[keys[keys.length - 1]] = value;
  return root;
}

/** Apply an AskUser suggestion choice to a manifest. `pose_pairing` is a
 * tagged-enum object keyed on `kind`; every other field takes the raw
 * choice value at its dotted path. The user can complete any remaining
 * sub-fields (e.g. the regex for shared_filename_token) in the form. */
export function applyAskUserChoice(manifest: unknown, field: string, choice: string): unknown {
  if (field === "pose_pairing") {
    return { ...(manifest as Record<string, unknown>), pose_pairing: { kind: choice } };
  }
  return setAtPath(manifest, field, choice);
}
