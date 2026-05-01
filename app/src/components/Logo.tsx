interface LogoProps {
  /** Pixel size of the bounding square. Logo scales uniformly. */
  size?: number;
  className?: string;
}

/* SVG geometry extracted from the vitavision logo
   (https://github.com/VitalyVorobyev/vitavision —
   src/components/shared/VitavisionLogo.tsx). Rendered as a static
   stroke-based "V" with a cyan dot pupil; no entrance animation —
   the diagnose viewer remounts the logo on every export reload, so
   a draw-in animation gets eye-poking fast. */
const V_OUTER =
  "m102.8 113.4 15.6-27.6h34.7l-40 71s-.6 1.3-3.3 4.7c-1.6 2-4.3 3-7 3q-4.2 0-7-2.9c-1.6-1.7-4-6.2-4-6.2L52.4 85.8H87z";
const V_INNER = "m102.6 112.3 28.2-15.1-28 49.5-28-49.5Z";

export function Logo({ size = 24, className }: LogoProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 107.9 82.9"
      xmlns="http://www.w3.org/2000/svg"
      shapeRendering="geometricPrecision"
      aria-label="vitavision V mark"
      className={className}
    >
      <g transform="translate(-48.8 -83.7)">
        <path
          d={V_OUTER}
          fill="none"
          stroke="currentColor"
          strokeWidth={5.4}
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <path
          d={V_INNER}
          fill="none"
          stroke="currentColor"
          strokeWidth={5.4}
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <circle cx={102.7} cy={112} r={11.8} fill="hsl(var(--background))" />
        <circle cx={102.6} cy={112} r={8.1} fill="hsl(var(--brand))" />
      </g>
    </svg>
  );
}
