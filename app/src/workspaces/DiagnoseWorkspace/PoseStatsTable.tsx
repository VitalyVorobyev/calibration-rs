import { cn, focusRing, Table, type Column } from "@vitavision/ui";
import { useMemo, useState } from "react";
import { colorForError } from "../../lib/errorColors";
import { computePoseResidualStats, type PoseResidualStat } from "../../lib/residualStats";
import type { TargetFeatureResidual } from "../../types";

type SortKey = "pose" | "count" | "mean" | "median" | "max";

interface PoseStatsTableProps {
  residuals: TargetFeatureResidual[];
  selectedPose: number;
  onSelectPose: (pose: number) => void;
}

const COLUMNS: { key: SortKey; label: string; numeric: boolean }[] = [
  { key: "pose", label: "pose", numeric: false },
  { key: "count", label: "n", numeric: true },
  { key: "mean", label: "mean", numeric: true },
  { key: "median", label: "med", numeric: true },
  { key: "max", label: "max", numeric: true },
];

/** Sort per-pose stats by one column. Pose sorts ascending by default;
 * the error/count columns sort descending (worst first) by default —
 * that's the useful reading for "which poses hurt". */
function sortStats(
  stats: PoseResidualStat[],
  key: SortKey,
  desc: boolean,
): PoseResidualStat[] {
  const sorted = [...stats].sort((a, b) => a[key] - b[key]);
  return desc ? sorted.reverse() : sorted;
}

/** Sortable per-pose reprojection stats (@vitavision/ui `Table`). Click a
 * header to sort, click a row (or Tab to it and press Enter) to jump the
 * viewer to that pose; the selected pose's row is `aria-current`. */
export function PoseStatsTable({
  residuals,
  selectedPose,
  onSelectPose,
}: PoseStatsTableProps) {
  const stats = useMemo(() => computePoseResidualStats(residuals), [residuals]);
  const [sortKey, setSortKey] = useState<SortKey>("mean");
  // Pose ascending, everything else descending (worst first) initially.
  const [desc, setDesc] = useState(true);

  const sorted = useMemo(() => sortStats(stats, sortKey, desc), [stats, sortKey, desc]);

  const onHeader = (key: SortKey) => {
    if (key === sortKey) {
      setDesc((v) => !v);
    } else {
      setSortKey(key);
      setDesc(key !== "pose");
    }
  };

  if (stats.length === 0) {
    return (
      <p className="text-[11px] text-fg-muted">
        No per-feature residuals in this export.
      </p>
    );
  }

  const columns: Column<PoseResidualStat>[] = COLUMNS.map((col) => ({
    key: col.key,
    numeric: col.numeric,
    header: (
      <button
        type="button"
        onClick={() => onHeader(col.key)}
        title={`Sort by ${col.label}`}
        className={cn(
          "cursor-pointer select-none rounded-control uppercase tracking-wider hover:text-fg",
          focusRing,
        )}
      >
        {col.label}
        {sortKey === col.key ? (desc ? " ↓" : " ↑") : ""}
      </button>
    ),
    cell: (s) => renderCell(col.key, s, s.pose === selectedPose),
  }));

  return (
    <section aria-label="Per-pose residual stats">
      <Table
        columns={columns}
        rows={sorted}
        rowKey={(s) => s.pose}
        onRowClick={(s) => onSelectPose(s.pose)}
        isRowActive={(s) => s.pose === selectedPose}
      />
    </section>
  );
}

function renderCell(key: SortKey, s: PoseResidualStat, active: boolean) {
  switch (key) {
    case "pose":
      return (
        <span
          className={cn("font-mono tabular-nums", active ? "text-signal" : "text-fg")}
        >
          {s.pose}
        </span>
      );
    case "count":
      return (
        <span className="text-fg-muted">
          {s.count}
          {s.diverged > 0 ? (
            <span
              className="ml-1 text-[10px] text-defect"
              title={`${s.diverged} diverged corner${s.diverged !== 1 ? "s" : ""}`}
            >
              +{s.diverged}✗
            </span>
          ) : null}
        </span>
      );
    default:
      return <ErrorCell value={s[key]} count={s.count} />;
  }
}

/** One px-error value, dotted with the shared severity color. Renders a
 * muted dash when the pose has no finite corners. */
function ErrorCell({ value, count }: { value: number; count: number }) {
  if (count === 0) return <span className="text-fg-muted">—</span>;
  return (
    <span className="inline-flex items-center justify-end gap-1.5">
      <span
        className="inline-block h-2 w-2 rounded-[2px]"
        style={{ background: colorForError(value) }}
        aria-hidden="true"
      />
      <span className="text-fg">{value.toFixed(3)}</span>
    </span>
  );
}
