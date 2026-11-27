import { ColormapLegend, colormapValue } from "@vitavision/charts";
import { Fragment, useMemo } from "react";
import type { ParameterCovariance } from "../../api/generated/types";
import { correlationMatrix } from "../../lib/uncertainty";

/** |ρ| of the parameters the covariance is over, on a perceptually uniform
 * map; each cell's title gives the signed value. */
export function CorrelationMatrix({ cov }: { cov: ParameterCovariance }) {
  const rho = useMemo(() => correlationMatrix(cov), [cov]);
  const n = cov.labels.length;
  if (n < 2) return null;
  return (
    <section aria-label="Parameter correlation" className="flex flex-col gap-1.5">
      <div
        className="grid items-center gap-x-1.5 font-mono text-[9px] text-fg-muted"
        style={{ gridTemplateColumns: "max-content 1fr" }}
      >
        {rho.map((row, i) => {
          const rowLabel = cov.labels[i]!;
          return (
            <Fragment key={rowLabel}>
              <span className="truncate text-right">{rowLabel}</span>
              <svg
                viewBox={`0 0 ${n} 1`}
                className="block w-full"
                role="img"
                shapeRendering="crispEdges"
              >
                <title>{`${rowLabel} correlations`}</title>
                {row.map((v, j) => (
                  <rect
                    key={cov.labels[j]}
                    x={j}
                    y={0}
                    width={1}
                    height={1}
                    fill={
                      Number.isFinite(v)
                        ? colormapValue("viridis", Math.abs(v), [0, 1])
                        : "none"
                    }
                  >
                    <title>{`${rowLabel} · ${cov.labels[j]}: ρ = ${
                      Number.isFinite(v) ? v.toFixed(3) : "—"
                    }`}</title>
                  </rect>
                ))}
              </svg>
            </Fragment>
          );
        })}
      </div>
      <ColormapLegend map="viridis" domain={[0, 1]} label="|ρ|" />
    </section>
  );
}
