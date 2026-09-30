interface HistogramProps {
  /** Bin counts (left-to-right). The component normalises by max. */
  bins: number[];
  /** 0-based bin index to highlight (cursor) — `null` for none. */
  cursorBin?: number | null;
  /** SVG width / height in CSS px. */
  width?: number;
  height?: number;
  className?: string;
}

/** Simple SVG histogram bar chart. Bars are grey-on-`--raised`; the
 * cursor bin is recoloured to `--signal` when supplied. The component
 * is presentational only — bin computation lives in the
 * `useImageData` hook. */
export function Histogram({
  bins,
  cursorBin,
  width = 240,
  height = 36,
  className,
}: HistogramProps) {
  const max = bins.reduce((m, v) => Math.max(m, v), 0);
  if (max === 0) {
    return (
      <svg width={width} height={height} className={className} aria-hidden>
        <rect width={width} height={height} fill="var(--raised)" />
      </svg>
    );
  }
  const barW = width / bins.length;
  return (
    <svg
      width={width}
      height={height}
      className={className}
      viewBox={`0 0 ${width} ${height}`}
      aria-label="grayscale histogram"
    >
      <rect width={width} height={height} fill="var(--raised)" />
      {bins.map((v, i) => {
        const h = (v / max) * (height - 2);
        const x = i * barW;
        const isCursor = cursorBin === i;
        return (
          <rect
            key={x}
            x={x}
            y={height - h}
            width={Math.max(barW - 0.5, 0.5)}
            height={h}
            fill={
              isCursor
                ? "var(--signal)"
                : "color-mix(in srgb, var(--fg-muted) 55%, transparent)"
            }
          />
        );
      })}
    </svg>
  );
}
