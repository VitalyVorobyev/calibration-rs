import { Badge, Button, Table, type Column } from "@vitavision/ui";
import { useMemo } from "react";
import { fixed } from "../../lib/format";
import { VERDICT } from "../../shell/runPills";
import { runKey, whenText, type UnitRow } from "./deviceModel";

/** A device type's units side by side: one row per unit with the run that
 * stands for it. Opening a unit shows its own history. */
export function UnitsTable({
  rows,
  activeKey,
  onHover,
  onOpenUnit,
}: {
  rows: UnitRow[];
  activeKey: string | null;
  onHover: (key: string | null) => void;
  onOpenUnit: (row: UnitRow) => void;
}) {
  const columns = useMemo(() => unitColumns(onOpenUnit), [onOpenUnit]);

  return (
    <Table
      caption="Units of the device type"
      columns={columns}
      rows={rows}
      rowKey={(r) => `${r.device.product ?? ""}\u0000${r.device.id}`}
      isRowActive={(r) =>
        r.representative !== null &&
        runKey(r.representative.project, r.representative.id) === activeKey
      }
      onRowHover={(r) =>
        onHover(
          r?.representative
            ? runKey(r.representative.project, r.representative.id)
            : null,
        )
      }
      onRowClick={(r) => onOpenUnit(r)}
      empty="No unit of this type has a run."
    />
  );
}

function unitColumns(onOpenUnit: (row: UnitRow) => void): Column<UnitRow>[] {
  return [
    {
      key: "unit",
      header: "Unit",
      width: "12rem",
      sortable: true,
      sortValue: (r) => r.device.id,
      cell: (r) => (
        <span className="flex min-w-0 flex-col">
          <span className="truncate font-medium text-fg">
            {r.device.label ?? r.device.id}
          </span>
          {r.device.label && (
            <span className="truncate font-mono text-[11px] text-fg-subtle">
              {r.device.id}
            </span>
          )}
        </span>
      ),
    },
    {
      key: "runs",
      header: "Runs",
      numeric: true,
      sortable: true,
      sortValue: (r) => r.runs,
      cell: (r) =>
        `${r.runs} in ${r.projects} ${r.projects === 1 ? "project" : "projects"}`,
    },
    {
      key: "starred",
      header: "Starred",
      numeric: true,
      sortable: true,
      sortValue: (r) => r.starred,
      cell: (r) => r.starred,
    },
    {
      key: "run",
      header: "Run",
      group: "Representative run",
      cell: (r) =>
        r.representative ? (
          <span className="flex flex-col">
            <span className="font-mono text-xs">{r.representative.id}</span>
            <span className="truncate text-[11px] text-fg-muted">
              {r.representative.project_name}
            </span>
          </span>
        ) : (
          <span className="text-xs text-fg-subtle">none: no starred or passing run</span>
        ),
    },
    {
      key: "verdict",
      header: "Verdict",
      group: "Representative run",
      cell: (r) => {
        const v = r.representative?.verdict;
        return v ? <Badge tone={VERDICT[v].tone}>{VERDICT[v].label}</Badge> : "–";
      },
    },
    {
      key: "rms",
      header: "RMS px",
      group: "Representative run",
      cell: (r) => {
        const cams = r.representative?.cameras ?? [];
        if (cams.length === 0) return "–";
        return (
          <span className="font-mono text-xs">
            {cams
              .map((c) => `${c.id} ${c.rms_px == null ? "–" : fixed(c.rms_px, 3)}`)
              .join(" · ")}
          </span>
        );
      },
    },
    {
      key: "when",
      header: "When",
      group: "Representative run",
      cell: (r) => (
        <span className="whitespace-nowrap font-mono text-xs text-fg-muted">
          {whenText(r.representative?.timestamp)}
        </span>
      ),
    },
    {
      key: "open",
      header: <span className="sr-only">Open</span>,
      width: "6.5rem",
      cell: (r) => (
        <span onClick={(e) => e.stopPropagation()} onKeyDown={(e) => e.stopPropagation()}>
          <Button
            size="sm"
            variant="secondary"
            aria-label={`Show the history of ${r.device.label ?? r.device.id}`}
            onClick={() => onOpenUnit(r)}
          >
            History
          </Button>
        </span>
      ),
    },
  ];
}
