import { Badge, Button, Table, type Column } from "@vitavision/ui";
import { Star } from "lucide-react";
import { useMemo } from "react";
import { fixed } from "../../lib/format";
import { VERDICT } from "../../shell/runPills";
import { cameraIds, rmsOf, seedLine, whenText, type HistoryRow } from "./deviceModel";

/** A device's runs in the library's order: where each came from, how it
 * judged, when, and the seed it started from. The representative run is
 * named, not only coloured; a run whose files cannot be read is a row with
 * its reason. Hovering a row marks the run in the trend charts, and a
 * click or Enter selects it. */
export function HistoryTable({
  rows,
  activeKey,
  onHover,
  onSelect,
  onOpen,
  compact = false,
}: {
  rows: HistoryRow[];
  /** The run hovered or selected, in every chart and here. */
  activeKey: string | null;
  onHover: (key: string | null) => void;
  onSelect: (key: string) => void;
  onOpen: (row: HistoryRow) => void;
  /** Only the run, its verdict and its time: the strip beside the charts. */
  compact?: boolean;
}) {
  const cameras = useMemo(() => cameraIds(rows), [rows]);
  const columns = useMemo(() => {
    const all = historyColumns(cameras, onOpen, compact);
    return compact ? all.filter((c) => COMPACT.includes(c.key)) : all;
  }, [cameras, onOpen, compact]);

  return (
    <Table
      caption="Runs of the device"
      columns={columns}
      rows={rows}
      rowKey={(r) => r.key}
      pinFirstColumn={!compact}
      isRowActive={(r) => r.key === activeKey}
      onRowHover={(r) => onHover(r ? r.key : null)}
      onRowClick={(r, _i, e) => {
        const double = e.type === "click" && (e as React.MouseEvent).detail >= 2;
        const enter = e.type === "keydown" && (e as React.KeyboardEvent).key === "Enter";
        if (double || enter) onOpen(r);
        else onSelect(r.key);
      }}
      empty="No run in this history."
    />
  );
}

const COMPACT = ["order", "run", "verdict", "when"];

/** The columns of the history: the run, its verdict, each camera's RMS,
 * when and the seed it started from. */
function historyColumns(
  cameras: readonly string[],
  onOpen: (row: HistoryRow) => void,
  compact: boolean,
): Column<HistoryRow>[] {
  return [
    {
      key: "order",
      header: "#",
      numeric: true,
      width: "3rem",
      sortable: true,
      sortValue: (r) => r.order + 1,
      cell: (r) => r.order + 1,
    },
    {
      key: "run",
      header: "Run",
      sortable: true,
      sortValue: (r) => `${r.run.project_name} ${r.run.id}`,
      cell: (r) => (
        <span className="flex min-w-0 flex-col gap-0.5">
          <span className="flex items-center gap-1.5 whitespace-nowrap font-mono text-xs">
            {r.run.starred && (
              <Star aria-label="starred" className="size-3 fill-current text-signal" />
            )}
            {r.run.id}
            {r.run.representative && <Badge tone="info">representative</Badge>}
          </span>
          <span className="truncate text-[11px] text-fg-muted" title={r.run.project}>
            {r.run.project_name}
            {r.run.label ? ` · ${r.run.label}` : ""}
          </span>
        </span>
      ),
    },
    {
      key: "verdict",
      header: "Verdict",
      sortable: true,
      sortValue: (r) => r.run.verdict ?? "",
      cell: (r) => <Outcome row={r} compact={compact} />,
    },
    ...cameras.map((id): Column<HistoryRow> => ({
      key: `rms:${id}`,
      header: `RMS ${id} px`,
      group: "Reprojection",
      numeric: true,
      sortable: true,
      sortValue: (r) => rmsOf(r.run, id),
      cell: (r) => {
        const v = rmsOf(r.run, id);
        return v === null ? "–" : fixed(v, 3);
      },
    })),
    {
      key: "when",
      header: "When",
      sortable: true,
      sortValue: (r) => r.run.timestamp,
      cell: (r) => (
        <span className="whitespace-nowrap font-mono text-xs text-fg-muted">
          {/* The strip beside the charts has room for the date only. */}
          {compact ? whenText(r.run.timestamp).slice(0, 10) : whenText(r.run.timestamp)}
        </span>
      ),
    },
    {
      key: "seed",
      header: "Started from",
      cell: (r) => {
        const line = seedLine(r.run.seed);
        return line ? (
          <span className="block max-w-[16rem] truncate text-xs" title={line.title}>
            {line.text}
          </span>
        ) : (
          <span className="text-xs text-fg-subtle">no seed</span>
        );
      },
    },
    {
      key: "open",
      header: <span className="sr-only">Open</span>,
      width: "5.5rem",
      cell: (r) => (
        <span onClick={(e) => e.stopPropagation()} onKeyDown={(e) => e.stopPropagation()}>
          <Button
            size="sm"
            variant="secondary"
            disabled={r.run.origin.kind === "unreadable"}
            aria-label={`Open ${r.run.id} of ${r.run.project_name} in Inspect`}
            onClick={() => onOpen(r)}
          >
            Inspect
          </Button>
        </span>
      ),
    },
  ];
}

/** The verdict in words; for a run that cannot be read, why. */
function Outcome({ row, compact }: { row: HistoryRow; compact: boolean }) {
  const origin = row.run.origin;
  if (origin.kind === "unreadable") {
    return (
      <span className="flex flex-col gap-0.5">
        <Badge tone="defect">unreadable</Badge>
        <span className="max-w-[14rem] text-[11px] text-fg-muted">{origin.message}</span>
      </span>
    );
  }
  const verdict = row.run.verdict;
  return (
    <span className="flex items-center gap-1.5">
      {verdict ? (
        <Badge tone={VERDICT[verdict].tone}>{VERDICT[verdict].label}</Badge>
      ) : (
        <span className="text-xs text-fg-subtle">not assessed</span>
      )}
      {!compact && origin.kind === "imported" && <Badge tone="neutral">imported</Badge>}
      {!compact && origin.kind === "unrecorded" && <Badge tone="neutral">legacy</Badge>}
    </span>
  );
}
