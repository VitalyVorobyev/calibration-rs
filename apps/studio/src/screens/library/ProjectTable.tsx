import {
  Badge,
  Button,
  DropdownMenu,
  HighlightedText,
  MenuItem,
  MenuSeparator,
  Table,
  toast,
  type Column,
} from "@vitavision/ui";
import { MoreHorizontal, Star } from "lucide-react";
import { useMemo } from "react";
import { api } from "../../api";
import { useElementWidth } from "../../hooks/useElementWidth";
import { fixed, formatCount } from "../../lib/format";
import { exportKindLabel } from "../../store/exportKind";
import { useProject } from "../../project/projectStore";
import { useReadOnly } from "../../session/jobs";
import { dateOf, querySpans, relativeToRoots, type ProjectRow } from "./libraryModel";
import { rowDetails, visibleColumns } from "./tableColumns";
import { VerdictBadge } from "./VerdictMark";

/** The projects of a tree node, one row each: name and place, workflow,
 * the run the row reports (the starred one, else the newest) with its
 * verdict and pooled RMS, the views it used, and when. Click selects;
 * double-click or Enter opens (on Inspect when there is a run to read, else
 * on Calibrate); the menu has the rest. */
export function ProjectTable({
  rows,
  roots,
  query,
  activeDir,
  onSelect,
  onOpen,
  onCalibrate,
}: {
  rows: ProjectRow[];
  roots: readonly string[];
  /** The search text, whose words are marked in the names. */
  query: string;
  /** The project the selection is on, when it is one. */
  activeDir: string | null;
  onSelect: (row: ProjectRow) => void;
  onOpen: (row: ProjectRow) => void;
  onCalibrate: (row: ProjectRow) => void;
}) {
  const readOnly = useReadOnly();
  const [ref, width] = useElementWidth<HTMLDivElement>();
  const columns = useMemo(() => {
    const keep = visibleColumns(width);
    return projectColumns({ query, roots, readOnly, onOpen, onCalibrate }).filter((c) =>
      keep.has(c.key),
    );
  }, [query, roots, readOnly, onOpen, onCalibrate, width]);

  return (
    <div ref={ref}>
      <Table
        caption="Projects"
        columns={columns}
        rows={rows}
        rowKey={(r) => r.entry.dir}
        pinFirstColumn
        defaultSort={{ key: "project", direction: "ascending" }}
        isRowActive={(r) => r.entry.dir === activeDir}
        onRowClick={(r, _i, e) => {
          const double = e.type === "click" && (e as React.MouseEvent).detail >= 2;
          const enter =
            e.type === "keydown" && (e as React.KeyboardEvent).key === "Enter";
          if (double || enter) onOpen(r);
          else onSelect(r);
        }}
        empty="No project here."
      />
    </div>
  );
}

interface ColumnContext {
  query: string;
  roots: readonly string[];
  readOnly: boolean;
  onOpen: (row: ProjectRow) => void;
  onCalibrate: (row: ProjectRow) => void;
}

function projectColumns({
  query,
  roots,
  readOnly,
  onOpen,
  onCalibrate,
}: ColumnContext): Column<ProjectRow>[] {
  return [
    {
      key: "project",
      header: "Name",
      sortable: true,
      sortValue: (r) => r.entry.name.toLowerCase(),
      cell: (r) => (
        <span className="flex min-w-0 flex-col" title={rowDetails(r)}>
          <HighlightedText
            text={r.entry.name}
            spans={querySpans(r.entry.name, query)}
            className="truncate font-medium text-fg"
          />
          <span
            className="truncate font-mono text-[11px] text-fg-subtle"
            title={r.entry.dir}
          >
            {relativeToRoots(r.entry.dir, roots)}
          </span>
        </span>
      ),
    },
    {
      key: "workflow",
      header: "Workflow",
      sortable: true,
      sortValue: (r) => r.entry.topology ?? "",
      cell: (r) =>
        r.entry.topology ? (
          <span className="whitespace-nowrap text-fg-muted">
            {exportKindLabel(r.entry.topology)}
          </span>
        ) : (
          <Badge tone="warning">manifest unreadable</Badge>
        ),
    },
    {
      key: "run",
      header: "Run",
      group: "Latest run",
      cell: (r) =>
        r.latest ? (
          <span className="flex items-center gap-1.5 font-mono text-xs">
            {r.latest.notes.starred && (
              <Star aria-label="starred" className="size-3 fill-current text-signal" />
            )}
            {r.latest.id}
            {r.entry.runs.length > 1 && (
              <span className="text-fg-subtle">of {r.entry.runs.length}</span>
            )}
          </span>
        ) : (
          <span className="text-xs text-fg-subtle">none</span>
        ),
    },
    {
      key: "verdict",
      header: "Verdict",
      group: "Latest run",
      sortable: true,
      sortValue: (r) =>
        ({ fail: 0, warn: 1, pass: 2 })[r.verdict ?? "pass"] + (r.verdict ? 0 : 3),
      cell: (r) => <VerdictBadge verdict={r.verdict} />,
    },
    {
      key: "rms",
      header: "RMS px",
      group: "Latest run",
      numeric: true,
      sortable: true,
      sortValue: (r) => r.rms,
      cell: (r) => (r.rms === null ? "–" : fixed(r.rms, 3)),
    },
    {
      key: "views",
      header: "Views",
      group: "Latest run",
      numeric: true,
      cell: (r) =>
        r.views ? `${formatCount(r.views.usable)}/${formatCount(r.views.total)}` : "–",
    },
    {
      key: "updated",
      header: "Updated",
      sortable: true,
      sortValue: (r) => r.updated,
      cell: (r) => (
        <span className="whitespace-nowrap font-mono text-xs text-fg-muted">
          {r.updated === null ? "–" : dateOf(r.updated)}
        </span>
      ),
    },
    {
      key: "actions",
      header: <span className="sr-only">Actions</span>,
      width: "7.5rem",
      cell: (r) => (
        <span
          className="flex items-center justify-end gap-1"
          // The row's own click selects; these are not it.
          onClick={(e) => e.stopPropagation()}
          onKeyDown={(e) => e.stopPropagation()}
        >
          <Button size="sm" variant="secondary" onClick={() => onOpen(r)}>
            Open
          </Button>
          <DropdownMenu
            align="end"
            trigger={
              <Button size="sm" variant="ghost" aria-label={`More for ${r.entry.name}`}>
                <MoreHorizontal aria-hidden className="size-4" />
              </Button>
            }
          >
            <MenuItem onSelect={() => onOpen(r)}>Open</MenuItem>
            <MenuItem onSelect={() => onCalibrate(r)}>Open in Calibrate</MenuItem>
            <MenuSeparator />
            <MenuItem
              onSelect={() => {
                void navigator.clipboard
                  ?.writeText(r.entry.dir)
                  .then(() =>
                    toast({ title: "Folder copied", description: r.entry.dir }),
                  );
              }}
            >
              Copy folder path
            </MenuItem>
            <MenuItem
              disabled={readOnly}
              onSelect={() => {
                void api
                  .forgetRecent({ dir: r.entry.dir })
                  .then(() => useProject.getState().refreshRecent());
              }}
            >
              Remove from recent
            </MenuItem>
          </DropdownMenu>
        </span>
      ),
    },
  ];
}
