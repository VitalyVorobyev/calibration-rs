/** Small manifest mutators used by the unresolved notice and the AskUser
 * modal. What a field means is its schema description, which the manifest
 * form shows. */

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
    const k = keys[i]!;
    if (typeof cur[k] !== "object" || cur[k] === null) cur[k] = {};
    cur = cur[k] as Record<string, unknown>;
  }
  cur[keys[keys.length - 1]!] = value;
  return root;
}

/** Apply an AskUser suggestion choice to a manifest. `pose_pairing` is a
 * tagged-enum object keyed on `kind`; every other field takes the raw
 * choice value at its dotted path. The user can complete any remaining
 * sub-fields (e.g. the regex for shared_filename_token) in the form. */
export function applyAskUserChoice(
  manifest: unknown,
  field: string,
  choice: string,
): unknown {
  if (field === "pose_pairing") {
    return { ...(manifest as Record<string, unknown>), pose_pairing: { kind: choice } };
  }
  return setAtPath(manifest, field, choice);
}
