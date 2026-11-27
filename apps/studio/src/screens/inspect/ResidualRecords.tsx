import { ScatterChart, type ScatterPick } from "@vitavision/charts";
import {
  Button,
  DensityProvider,
  ErrorBox,
  Skeleton,
  Table,
  type Column,
} from "@vitavision/ui";
import { useMemo, useState } from "react";
import { api } from "../../api";
import type { RunRef } from "../../api/generated/types";
import { fixed } from "../../lib/format";
import { useQuery } from "../../session/query";
import { useSelection } from "../../session/selection";
import { equalAxis, recordRows, scatterGroups, type RecordRow } from "./residualModel";

const px = (v: number) => fixed(v, 3);

const COLUMNS: Column<RecordRow>[] = [
  {
    key: "view",
    header: "View",
    numeric: true,
    width: "4rem",
    sortValue: (r) => r.view,
    cell: (r) => r.view,
  },
  {
    key: "feature",
    header: "Feature",
    numeric: true,
    width: "5rem",
    sortValue: (r) => r.feature,
    cell: (r) => r.feature,
  },
  {
    key: "x",
    header: "x (px)",
    numeric: true,
    width: "5rem",
    sortValue: (r) => r.x,
    cell: (r) => fixed(r.x, 1),
  },
  {
    key: "y",
    header: "y (px)",
    numeric: true,
    width: "5rem",
    sortValue: (r) => r.y,
    cell: (r) => fixed(r.y, 1),
  },
  {
    key: "dx",
    header: "dx (px)",
    numeric: true,
    width: "5.5rem",
    sortValue: (r) => r.dx,
    cell: (r) => px(r.dx),
  },
  {
    key: "dy",
    header: "dy (px)",
    numeric: true,
    width: "5.5rem",
    sortValue: (r) => r.dy,
    cell: (r) => px(r.dy),
  },
  {
    key: "magnitude",
    header: "|r| (px)",
    numeric: true,
    width: "5.5rem",
    firstSort: "descending",
    sortValue: (r) => r.magnitude,
    cell: (r) => px(r.magnitude),
  },
  {
    key: "outlier",
    header: "Outlier",
    width: "5rem",
    sortValue: (r) => (r.outlier ? 1 : 0),
    cell: (r) => (r.outlier ? "yes" : ""),
  },
];

/** Camera `camera`'s residuals as a scatter of `observed − projected` and a
 * table of the same records, the worst first. Choosing a point or a row
 * selects its view and feature, which the frame viewer follows. `views` is
 * the scope the map is over; `outliers` are the map's left-out records. */
export function ResidualRecords({
  run,
  camera,
  views,
  outliers,
}: {
  run: RunRef;
  camera: number;
  views: number[] | null;
  outliers: readonly number[];
}) {
  const q = useQuery("residual_records", { run, camera }, () =>
    api.residualRecords({ run, camera }),
  );
  const selection = useSelection((s) => s.selection);
  const outlierSet = useMemo(() => new Set(outliers), [outliers]);
  const { rows, diverged } = useMemo(
    () =>
      q.data
        ? recordRows(q.data, outlierSet, views)
        : { rows: [] as RecordRow[], diverged: 0 },
    [q.data, outlierSet, views],
  );
  const groups = useMemo(() => scatterGroups(rows), [rows]);
  const axis = useMemo(() => equalAxis(rows), [rows]);
  const [hover, setHover] = useState<RecordRow | null>(null);

  const choose = (row: RecordRow) => {
    const s = useSelection.getState();
    s.selectCamera(camera);
    s.selectView(row.view);
    s.selectFeature(row.feature);
  };
  const isChosen = (r: RecordRow) =>
    selection.view === r.view && selection.feature === r.feature;
  const chosen = rows.find(isChosen) ?? hover;
  const highlighted = chosen
    ? (() => {
        const g = groups.find((gr) => gr.rows.includes(chosen));
        return g ? { series: g.name, index: g.rows.indexOf(chosen) } : null;
      })()
    : null;
  const pick = (p: ScatterPick | null): RecordRow | null =>
    p ? (groups.find((g) => g.name === p.series)?.rows[p.index] ?? null) : null;

  if (q.error && !q.data) {
    return (
      <ErrorBox>
        Could not read the residuals: {q.error.message}{" "}
        <Button size="sm" onClick={q.refetch}>
          Retry
        </Button>
      </ErrorBox>
    );
  }
  if (!q.data) return <Skeleton className="h-64 w-full" />;
  if (rows.length === 0) {
    return (
      <p className="text-sm text-fg-muted">
        This run holds no per-feature residuals for camera {camera}
        {views ? " in the chosen views" : ""}.
      </p>
    );
  }
  return (
    <div className="grid min-h-0 gap-4 lg:grid-cols-5">
      <section aria-label="Residual scatter" className="min-w-0 lg:col-span-2">
        <ScatterChart
          label={`Residuals of camera ${camera}: dy against dx`}
          variant="fluid"
          height={280}
          xLabel="dx = observed − projected (px)"
          yLabel="dy (px)"
          xDomain={axis}
          yDomain={axis}
          series={groups.map((g) => ({
            name: g.name,
            points: g.rows.map((r) => ({ x: r.dx, y: r.dy })),
          }))}
          highlighted={highlighted}
          onHover={(p) => setHover(pick(p))}
          onPick={(p) => {
            const row = pick(p);
            if (row) choose(row);
          }}
          referenceLines={[
            { axis: "x", value: 0 },
            { axis: "y", value: 0 },
          ]}
        />
      </section>
      <section
        aria-label="Residual records"
        className="flex min-w-0 flex-col gap-1 lg:col-span-3"
      >
        <p className="text-xs text-fg-muted">
          {rows.length} features{diverged > 0 ? `, ${diverged} without a projection` : ""}
          , the worst first. A row selects its view and feature.
        </p>
        <DensityProvider value="compact">
          <Table
            caption="Residual records"
            columns={COLUMNS}
            rows={rows}
            rowKey={(r) => r.index}
            defaultSort={{ key: "magnitude", direction: "descending" }}
            windowed={{ rowHeight: 24, height: 280 }}
            onRowClick={choose}
            isRowActive={isChosen}
            onRowHover={(r) => setHover(r)}
          />
        </DensityProvider>
      </section>
    </div>
  );
}
