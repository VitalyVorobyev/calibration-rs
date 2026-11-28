/** What the map's blank and faded areas mean, beside its colour legend: a
 * blank node has no value (it is not zero), a faded one has a value that is
 * not distinguishable from zero (or, where the map says so, lies outside
 * what a fit covers). */
export function LegendKey({
  faded,
  fadedText = "not distinguishable from zero (p ≥ 0.01)",
}: {
  faded: boolean;
  /** What a faded node is, where it is not a significance test. */
  fadedText?: string | undefined;
}) {
  return (
    <ul aria-label="Map key" className="flex flex-col gap-1.5 text-xs text-fg-muted">
      <li className="flex items-center gap-2">
        <span
          aria-hidden="true"
          className="inline-block size-3.5 shrink-0 rounded-[2px] border border-dashed border-line-strong bg-canvas"
        />
        no data
      </li>
      {faded && (
        <li className="flex items-center gap-2">
          <span
            aria-hidden="true"
            className="inline-block size-3.5 shrink-0 rounded-[2px] bg-signal opacity-25"
          />
          <span>{fadedText}</span>
        </li>
      )}
    </ul>
  );
}
