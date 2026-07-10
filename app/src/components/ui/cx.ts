/** Tiny conditional-className joiner — the same `.filter(Boolean).join(" ")`
 * idiom several workspace files already hand-rolled (RailLink,
 * CollapsibleSection, PresetCard, …), pulled into one place instead of a
 * `clsx`/`classnames` dependency (none was needed for this small a need). */
export type ClassValue = string | false | null | undefined;

export function cx(...values: ClassValue[]): string {
  return values.filter(Boolean).join(" ");
}
