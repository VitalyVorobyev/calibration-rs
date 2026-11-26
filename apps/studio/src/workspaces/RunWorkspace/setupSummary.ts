/** One-line summaries of the Run setup, for the collapsed form sections
 * and the header subtitle. */

/** The manifest's target kind, `?` when unset. */
export function targetKindOf(manifest: unknown): string {
  const m = manifest as Record<string, unknown> | null;
  const target = m?.target as Record<string, unknown> | undefined;
  return typeof target?.kind === "string" ? target.kind : "?";
}

/** `topology · target · N cameras`. */
export function manifestSummary(manifest: unknown): string {
  const m = manifest as Record<string, unknown> | null;
  const topology = typeof m?.topology === "string" ? m.topology : "?";
  const cameras = Array.isArray(m?.cameras) ? m.cameras.length : 0;
  return `${topology} · ${targetKindOf(manifest)} · ${cameras} camera${cameras !== 1 ? "s" : ""}`;
}

/** `max_iters=N · loss=L`: the solver iteration cap and the reprojection
 * loss of every stage (one name when they agree, `a/b` when they differ). */
export function configSummary(config: unknown): string {
  const c = config as Record<string, unknown> | null;
  const solver = c?.solver as Record<string, unknown> | undefined;
  const iters = typeof solver?.max_iters === "number" ? solver.max_iters : "?";
  const kinds = [...new Set(reprojectionLosses(c).map(robustLossLabel))];
  return `max_iters=${iters} · loss=${kinds.length > 0 ? kinds.join("/") : "none"}`;
}

/** Every `reprojection_loss` value nested anywhere in a config. */
function reprojectionLosses(value: unknown): unknown[] {
  if (!value || typeof value !== "object" || Array.isArray(value)) return [];
  const out: unknown[] = [];
  for (const [key, child] of Object.entries(value)) {
    if (key === "reprojection_loss") out.push(child);
    else out.push(...reprojectionLosses(child));
  }
  return out;
}

/** A robust-loss config value's name: the `kind` tag of
 * `{ "kind": "huber", "scale": … }`; `none` when absent. */
export function robustLossLabel(value: unknown): string {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    const kind = (value as Record<string, unknown>).kind;
    if (typeof kind === "string") return kind;
  }
  return "none";
}

/** Why Run is blocked, or `undefined` when it is not. */
export function runBlockReason(opts: {
  supported: boolean;
  unsupportedReason?: string | undefined;
  unresolved: number;
  manifestDir: string | null;
}): string | undefined {
  if (!opts.supported) return opts.unsupportedReason;
  if (opts.unresolved > 0) {
    const n = opts.unresolved;
    return `Resolve ${n} unresolved manifest field${n !== 1 ? "s" : ""} first`;
  }
  if (!opts.manifestDir) return "Set a dataset folder first";
  return undefined;
}
