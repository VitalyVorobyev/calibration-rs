/** A configuration patch as TOML, the way a project file states it: one
 * dotted key per leaf (`solver.max_iters = 20`). A list of tables is a
 * list of inline tables. TOML has no `null`: a leaf that sets a field back
 * to none is written as a comment, since only JSON can say it. */

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

const BARE_KEY = /^[A-Za-z0-9_-]+$/;

function key(name: string): string {
  return BARE_KEY.test(name) ? name : JSON.stringify(name);
}

function value(v: unknown): string {
  if (typeof v === "string") return JSON.stringify(v);
  if (typeof v === "number" || typeof v === "boolean") return String(v);
  if (Array.isArray(v)) return `[${v.map(value).join(", ")}]`;
  if (isRecord(v)) {
    const entries = Object.entries(v).filter(([, x]) => x !== null);
    return entries.length === 0
      ? "{}"
      : `{ ${entries.map(([k, x]) => `${key(k)} = ${value(x)}`).join(", ")} }`;
  }
  return "{}";
}

/** The lines of `patch`, in key order. Empty for an empty patch. */
export function patchToml(patch: unknown, prefix: string[] = []): string[] {
  if (!isRecord(patch)) return [];
  const lines: string[] = [];
  for (const [name, v] of Object.entries(patch).sort(([a], [b]) => (a < b ? -1 : 1))) {
    const path = [...prefix, name];
    const dotted = path.map(key).join(".");
    if (v === null) {
      lines.push(`# ${dotted} = none (JSON only)`);
    } else if (isRecord(v) && Object.keys(v).length > 0) {
      lines.push(...patchToml(v, path));
    } else {
      lines.push(`${dotted} = ${value(v)}`);
    }
  }
  return lines;
}
