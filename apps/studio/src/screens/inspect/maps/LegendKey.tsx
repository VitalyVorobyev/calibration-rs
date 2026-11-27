/** What the map's blank and faded areas mean, beside its colour legend: a
 * blank node has no value (it is not zero), a faded one has a value that is
 * not distinguishable from zero. */
export function LegendKey({ faded }: { faded: boolean }) {
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
          <span>not distinguishable from zero (p ≥ 0.01)</span>
        </li>
      )}
    </ul>
  );
}
