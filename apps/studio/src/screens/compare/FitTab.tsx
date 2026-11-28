import { Badge, DeltaCell, Table, type Column } from "@vitavision/ui";
import { useMemo, type ReactNode } from "react";
import type {
  CameraEvidence,
  RunComparison,
  RunEvidence,
  Verdict,
} from "../../api/generated/types";
import { fixed, formatCount } from "../../lib/format";
import { VERDICT } from "../../shell/runPills";
import {
  evidenceCamera,
  maxProjectionSigma,
  pooledRms,
  positionName,
  residualsComparable,
  signed,
} from "./compareModel";
import { covarianceWords } from "./compareWords";

const px = (v: number | null | undefined) => (v == null ? "–" : fixed(v, 3));

interface RunRow {
  index: number;
  run: RunEvidence;
  verdict: Verdict;
}

interface CameraRow {
  index: number;
  name: string;
  camera: CameraEvidence | null;
}

/** How well each run fits its own observations, side by side: the quality
 * verdict and covariance of each run, then per camera the views and
 * features it used and its residuals. A residual compares with the
 * reference's only for runs of the same observations; the tab says where
 * it does not. */
export function FitTab({
  comparison,
  verdicts,
}: {
  comparison: RunComparison;
  /** Each run's verdict as recorded with it, where there is one. */
  verdicts: (Verdict | null)[];
}) {
  const runs = comparison.runs;
  const cameras = useMemo(() => {
    const ids: string[] = [];
    for (const r of runs)
      for (const c of r.cameras) if (!ids.includes(c.id)) ids.push(c.id);
    return ids;
  }, [runs]);
  const caveats = comparison.differences
    .filter((d) => !residualsComparable(d.relation.kind))
    .map((d) => runs[d.run]?.name ?? "");

  return (
    <div className="flex flex-col gap-4">
      <RunsTable runs={runs} verdicts={verdicts} />
      {caveats.length > 0 && (
        <p className="max-w-3xl text-xs text-fg-muted">
          Residuals compare like with like only between runs of the same observations.{" "}
          {caveats.join(", ")} did not use the observations of {runs[0]?.name}: a run that
          left out its worst features always has a lower residual, which says nothing of
          its calibration, so no change in RMS is shown for{" "}
          {caveats.length === 1 ? "it" : "them"}.
        </p>
      )}
      {cameras.map((id) => (
        <CameraFit key={id} id={id} comparison={comparison} />
      ))}
    </div>
  );
}

const RUN_COLUMNS: Column<RunRow>[] = [
  {
    key: "run",
    header: "Run",
    cell: (r) => (
      <span className="flex items-center gap-1.5 font-mono text-xs">
        <Badge tone={r.index === 0 ? "info" : "neutral"}>{positionName(r.index)}</Badge>
        {r.run.name}
      </span>
    ),
  },
  {
    key: "verdict",
    header: "Quality",
    cell: (r) => <Badge tone={VERDICT[r.verdict].tone}>{VERDICT[r.verdict].label}</Badge>,
  },
  {
    key: "rms",
    header: "RMS px",
    numeric: true,
    sortable: true,
    sortValue: (r) => pooledRms(r.run),
    cell: (r) => px(pooledRms(r.run)),
  },
  {
    key: "sigma",
    header: "Largest projection σ px",
    numeric: true,
    sortable: true,
    sortValue: (r) => maxProjectionSigma(r.run),
    cell: (r) => px(maxProjectionSigma(r.run)),
  },
  {
    key: "cov",
    header: "Covariance",
    cell: (r) => {
      const why = covarianceWords(r.run.covariance);
      const c = r.run.covariance;
      return why ? (
        <span className="text-xs text-fg-muted" title={why}>
          unavailable: {why}
        </span>
      ) : (
        <span className="text-xs">
          available
          {c.status === "available" && c.view_scatter != null
            ? ` · view scatter ×${fixed(c.view_scatter, 2)}`
            : ""}
        </span>
      );
    },
  },
];

function RunsTable({
  runs,
  verdicts,
}: {
  runs: RunEvidence[];
  verdicts: (Verdict | null)[];
}) {
  const rows = useMemo(
    () =>
      runs.map((run, index): RunRow => ({
        index,
        run,
        verdict: verdicts[index] ?? run.verdict,
      })),
    [runs, verdicts],
  );
  return (
    <section aria-label="The runs" className="flex flex-col gap-1.5">
      <h3 className="text-sm text-fg">Runs</h3>
      <Table
        caption="The runs' quality and uncertainty"
        columns={RUN_COLUMNS}
        rows={rows}
        rowKey={(r) => r.run.name}
      />
    </section>
  );
}

/** The columns of one camera's table; `rms` writes a run's RMS (with its
 * change from the reference where the runs compare). */
function cameraColumns(rms: (row: CameraRow) => ReactNode): Column<CameraRow>[] {
  const stat = (
    key: string,
    header: string,
    group: string,
    value: (c: CameraEvidence) => number | null | undefined,
  ): Column<CameraRow> => ({
    key,
    header,
    group,
    numeric: true,
    cell: (r) => (r.camera ? px(value(r.camera)) : "–"),
  });
  const count = (
    key: string,
    header: string,
    value: (c: CameraEvidence) => number,
  ): Column<CameraRow> => ({
    key,
    header,
    numeric: true,
    cell: (r) => (r.camera ? formatCount(value(r.camera)) : "–"),
  });
  return [
    {
      key: "run",
      header: "Run",
      cell: (r) => (
        <span className="flex items-center gap-1.5 font-mono text-xs">
          <Badge tone={r.index === 0 ? "info" : "neutral"}>{positionName(r.index)}</Badge>
          {r.name}
        </span>
      ),
    },
    count("views", "Views", (c) => c.views),
    count("features", "Features", (c) => c.features),
    count("unprojected", "Unprojected", (c) => c.unprojected),
    {
      key: "rms",
      header: "RMS px",
      group: "Reprojection",
      numeric: true,
      sortable: true,
      sortValue: (r) => r.camera?.reprojection?.rms ?? null,
      cell: rms,
    },
    stat("mean", "Mean px", "Reprojection", (c) => c.reprojection?.mean),
    stat("median", "Median px", "Reprojection", (c) => c.reprojection?.median),
    stat("p95", "95th px", "Reprojection", (c) => c.reprojection?.p95),
    stat("max", "Max px", "Reprojection", (c) => c.reprojection?.max),
    stat("psigma", "Largest σ px", "Projection", (c) => c.max_projection_sigma_px),
  ];
}

/** A run's RMS, with its change from the reference when both are of the
 * same observations. */
function rmsCell(
  row: CameraRow,
  reference: number | null,
  comparable: boolean,
): ReactNode {
  const v = row.camera?.reprojection?.rms;
  if (v == null) return "–";
  if (!comparable || reference === null) return fixed(v, 3);
  return (
    <DeltaCell
      value={v}
      delta={v - reference}
      magnitude={reference > 0 ? Math.abs(v - reference) / reference / 0.2 : null}
      tone="neutral"
      format={(x) => fixed(x, 3)}
      deltaFormat={(d) => signed(d).replace(/^[+−]/, "")}
      deltaLabel="change from the reference"
    />
  );
}

function CameraFit({ id, comparison }: { id: string; comparison: RunComparison }) {
  const rows = useMemo(
    () =>
      comparison.runs.map((run, index): CameraRow => ({
        index,
        name: run.name,
        camera: evidenceCamera(run, id),
      })),
    [comparison, id],
  );
  const columns = useMemo(() => {
    const reference = rows[0]?.camera?.reprojection?.rms ?? null;
    return cameraColumns((r) =>
      rmsCell(
        r,
        reference,
        r.index > 0 &&
          residualsComparable(
            comparison.differences[r.index - 1]?.relation.kind ?? "unknown",
          ),
      ),
    );
  }, [rows, comparison]);
  return (
    <section aria-label={`Fit of ${id}`} className="flex flex-col gap-1.5">
      <h3 className="font-mono text-sm text-fg">{id}</h3>
      <Table
        caption={`Fit quality of ${id}`}
        columns={columns}
        rows={rows}
        rowKey={(r) => r.name}
        pinFirstColumn
      />
    </section>
  );
}
