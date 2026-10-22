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

/** `max_iters=N · loss=L`. */
export function configSummary(config: unknown): string {
  const c = config as Record<string, unknown> | null;
  const iters = typeof c?.max_iters === "number" ? c.max_iters : "?";
  return `max_iters=${iters} · loss=${robustLossLabel(c?.robust_loss)}`;
}

/** A robust-loss config value's name: a bare string, a `{ type }` tag or
 * an externally tagged `{ Name: … }`. */
export function robustLossLabel(value: unknown): string {
  if (typeof value === "string") return value;
  if (value && typeof value === "object" && !Array.isArray(value)) {
    const obj = value as Record<string, unknown>;
    if (typeof obj.type === "string") return obj.type;
    const key = Object.keys(obj)[0];
    if (key) return key;
  }
  return "None";
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
