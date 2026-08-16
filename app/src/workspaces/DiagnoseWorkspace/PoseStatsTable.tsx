import { useMemo, useState } from "react";
import { colorForError } from "../../components/FrameCanvas";
import { Table, Td, Th } from "../../components/ui";
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

/** Sortable per-pose reprojection stats. Click a header to sort, click a
 * row to jump the viewer to that pose. */
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
      <p className="text-[11px] text-muted-foreground">
        No per-feature residuals in this export.
      </p>
    );
  }

  return (
    <section aria-label="Per-pose residual stats">
      <Table className="w-full">
        <thead>
          <tr className="border-b border-border text-muted-foreground">
            {COLUMNS.map((col) => (
              <Th
                key={col.key}
                align={col.numeric ? "right" : "left"}
                onClick={() => onHeader(col.key)}
                className="cursor-pointer select-none py-1 uppercase tracking-wider hover:text-foreground"
                title={`Sort by ${col.label}`}
              >
                {col.label}
                {sortKey === col.key ? (desc ? " ↓" : " ↑") : ""}
              </Th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((s) => {
            const active = s.pose === selectedPose;
            return (
              <tr
                key={s.pose}
                onClick={() => onSelectPose(s.pose)}
                className={`cursor-pointer border-b border-border/50 transition-colors hover:bg-bg-soft ${
                  active ? "bg-brand/[0.08]" : ""
                }`}
                title={`Jump to pose ${s.pose}`}
              >
                <Td className={`py-1 ${active ? "text-brand" : "text-foreground"}`}>
                  {s.pose}
                </Td>
                <Td align="right" className="py-1 text-muted-foreground">
                  {s.count}
                  {s.diverged > 0 ? (
                    <span
                      className="ml-1 text-[10px] text-destructive"
                      title={`${s.diverged} diverged corner${s.diverged !== 1 ? "s" : ""}`}
                    >
                      +{s.diverged}✗
                    </span>
                  ) : null}
                </Td>
                <Td align="right" className="py-1">
                  <ErrorCell value={s.mean} count={s.count} />
                </Td>
                <Td align="right" className="py-1">
                  <ErrorCell value={s.median} count={s.count} />
                </Td>
                <Td align="right" className="py-1">
                  <ErrorCell value={s.max} count={s.count} />
                </Td>
              </tr>
            );
          })}
        </tbody>
      </Table>
    </section>
  );
}

/** One px-error value, dotted with the shared severity color. Renders a
 * muted dash when the pose has no finite corners. */
function ErrorCell({ value, count }: { value: number; count: number }) {
  if (count === 0) return <span className="text-muted-foreground">—</span>;
  return (
    <span className="inline-flex items-center justify-end gap-1.5">
      <span
        className="inline-block h-2 w-2 rounded-[2px]"
        style={{ background: colorForError(value) }}
        aria-hidden="true"
      />
      <span className="text-foreground">{value.toFixed(3)}</span>
    </span>
  );
}
