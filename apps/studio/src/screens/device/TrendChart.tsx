import { LineChart } from "@vitavision/charts";
import { useMemo } from "react";
import { fixed } from "../../lib/format";
import type { XMode } from "./deviceModel";
import {
  chartSeries,
  gapNote,
  markersFor,
  nearestRun,
  spreadLine,
  xAxisLabel,
  xFormatter,
  xOfRun,
  type ChartSpec,
  type RunMeta,
} from "./seriesModel";

/** One parameter over the runs: a line per unit with its σ band, gaps where
 * a run lacks it, ticks for the representative, starred and at-bound runs,
 * and the library's spread of the values beside it. Hovering a run here
 * marks it in every other chart and in the history table. */
export function TrendChart({
  spec,
  xMode,
  meta,
  unitIndex,
  hoverKey,
  onHover,
  onPick,
}: {
  spec: ChartSpec;
  xMode: XMode;
  meta: ReadonlyMap<string, RunMeta>;
  /** Each unit's colour slot, the same in every chart. */
  unitIndex: ReadonlyMap<string, number>;
  hoverKey: string | null;
  onHover: (key: string | null) => void;
  onPick: (key: string) => void;
}) {
  const many = spec.lines.length > 1;
  const series = useMemo(
    () =>
      spec.lines.map((line) =>
        chartSeries(
          line,
          unitIndex.get(line.unit) ?? 0,
          many ? line.unit : spec.name,
          spec.sigmaOf,
        ),
      ),
    [spec, unitIndex, many],
  );
  const markers = useMemo(() => markersFor(spec.lines, meta), [spec.lines, meta]);
  const drawn = spec.lines.some((l) => l.entries.some((e) => !Number.isNaN(e.y)));
  const unit = spec.unit ? ` (${spec.unit})` : "";

  return (
    <section
      aria-label={`${spec.key} over the runs`}
      className="flex min-w-0 flex-col gap-2 rounded-panel border border-line bg-surface p-3"
    >
      <header className="flex items-baseline justify-between gap-2">
        <h3 className="font-mono text-sm font-medium text-fg">
          {spec.name}
          <span className="font-sans text-xs font-normal text-fg-muted">{unit}</span>
        </h3>
        {spec.sigmaOf === "perturbation" && (
          <span
            className="text-[11px] text-fg-muted"
            title="The band is the σ of a small correction of this pose, not of the pose's value."
          >
            σ of a perturbation
          </span>
        )}
      </header>
      {drawn ? (
        <LineChart
          label={`${spec.key} over the runs`}
          variant="fluid"
          height={150}
          xLabel={xAxisLabel(xMode)}
          yLabel={spec.unit || spec.name}
          xFormat={xFormatter(xMode)}
          series={series}
          markers={markers}
          cursor={xOfRun(spec.lines, hoverKey)}
          onHover={(x) => onHover(x === null ? null : nearestRun(spec.lines, x))}
          onPick={(x) => {
            const key = nearestRun(spec.lines, x);
            if (key) onPick(key);
          }}
        />
      ) : (
        <p className="grid h-24 place-items-center rounded-control bg-raised text-xs text-fg-muted">
          No run in view has this value.
        </p>
      )}
      <ul className="flex flex-col gap-0.5 text-[11px] text-fg-muted">
        {spec.lines.map((line) => {
          const note = gapNote(line, xMode);
          return note ? (
            <li key={line.unit}>
              {many ? `${line.unit}: ` : ""}
              {note}
            </li>
          ) : null;
        })}
      </ul>
      <Stats spec={spec} />
    </section>
  );
}

/** The library's spread of the values, or why there is none. */
function Stats({ spec }: { spec: ChartSpec }) {
  return (
    <dl className="flex flex-col gap-1 border-t border-line pt-2 text-[11px]">
      {spec.lines.map((line) => (
        <div key={line.unit} className="flex flex-col gap-0.5">
          {spec.lines.length > 1 && <dt className="font-medium text-fg">{line.unit}</dt>}
          <dd className="text-fg-muted">
            {line.stats.kind === "spread" ? (
              <span
                className="font-mono"
                title="Over the runs that make a seed: median, 1.4826 · MAD, min, max."
              >
                {spreadLine(line.stats.spread)} · n {fixed(line.stats.runs, 0)}
              </span>
            ) : (
              <span>{line.stats.reason}</span>
            )}
          </dd>
        </div>
      ))}
    </dl>
  );
}
