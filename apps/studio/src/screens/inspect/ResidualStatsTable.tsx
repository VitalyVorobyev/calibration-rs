import {
  Button,
  DensityProvider,
  ErrorBox,
  SegmentedControl,
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
import { statsRows, type StatsRow, type StatsScope } from "./residualModel";

const COLUMNS: Column<StatsRow>[] = [
  { key: "label", header: "Scope", sortValue: (r) => r.label, cell: (r) => r.label },
  {
    key: "count",
    header: "Features",
    numeric: true,
    sortValue: (r) => r.count,
    cell: (r) => r.count,
  },
  {
    key: "mean",
    header: "Mean (px)",
    numeric: true,
    sortValue: (r) => r.mean,
    cell: (r) => fixed(r.mean, 3),
  },
  {
    key: "median",
    header: "Median (px)",
    numeric: true,
    sortValue: (r) => r.median ?? -1,
    cell: (r) => (r.median === null ? "–" : fixed(r.median, 3)),
  },
  {
    key: "rms",
    header: "RMS (px)",
    numeric: true,
    firstSort: "descending",
    sortValue: (r) => r.rms,
    cell: (r) => fixed(r.rms, 3),
  },
  {
    key: "p95",
    header: "P95 (px)",
    numeric: true,
    firstSort: "descending",
    sortValue: (r) => r.p95 ?? -1,
    cell: (r) => (r.p95 === null ? "–" : fixed(r.p95, 3)),
  },
  {
    key: "max",
    header: "Max (px)",
    numeric: true,
    firstSort: "descending",
    sortValue: (r) => r.max,
    cell: (r) => fixed(r.max, 3),
  },
  {
    key: "unprojected",
    header: "No projection",
    numeric: true,
    sortValue: (r) => r.unprojected,
    cell: (r) => r.unprojected,
  },
];

/** The run's reprojection statistics per camera, per view or per camera and
 * view. A row selects its camera and view, which the frame viewer shows. */
export function ResidualStatsTable({ run }: { run: RunRef }) {
  const q = useQuery("residual_stats", { run }, () => api.residualStats({ run }));
  const [scope, setScope] = useState<StatsScope>("view");
  const selection = useSelection((s) => s.selection);
  const rows = useMemo(() => (q.data ? statsRows(q.data, scope) : []), [q.data, scope]);

  if (q.error && !q.data) {
    return (
      <ErrorBox>
        Could not read the statistics: {q.error.message}{" "}
        <Button size="sm" onClick={q.refetch}>
          Retry
        </Button>
      </ErrorBox>
    );
  }
  if (!q.data) return <Skeleton className="h-40 w-full" />;
  return (
    <section
      aria-label="Reprojection statistics"
      className="flex min-w-0 flex-col gap-1.5"
    >
      <div className="flex items-center gap-3">
        <SegmentedControl
          aria-label="Statistics per"
          value={scope}
          onValueChange={(v) => setScope((v || "view") as StatsScope)}
          options={[
            { value: "camera", label: "Camera" },
            { value: "view", label: "View" },
            { value: "cell", label: "Camera × view" },
          ]}
        />
        <span className="text-xs text-fg-muted">|observed − projected|, in pixels</span>
      </div>
      <DensityProvider value="compact">
        <Table
          caption="Reprojection statistics"
          columns={COLUMNS}
          rows={rows}
          rowKey={(r) => r.key}
          empty="The run has no per-feature residuals to summarise."
          onRowClick={(r) => {
            const s = useSelection.getState();
            if (r.camera !== null) s.selectCamera(r.camera);
            if (r.view !== null) s.selectView(r.view);
          }}
          isRowActive={(r) =>
            (r.view === null || r.view === selection.view) &&
            (r.camera === null || r.camera === selection.camera) &&
            (r.view !== null || r.camera !== null)
          }
          windowed={{ rowHeight: 24, height: 240 }}
        />
      </DensityProvider>
    </section>
  );
}
