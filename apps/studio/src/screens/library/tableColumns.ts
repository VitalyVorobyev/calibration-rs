/** Which of the project table's columns a width has room for, and what the
 * dropped ones say of a row. */
import { formatCount } from "../../lib/format";
import { exportKindLabel } from "../../store/exportKind";
import { dateOf, type ProjectRow } from "./libraryModel";

const ALL_COLUMNS = [
  "project",
  "workflow",
  "run",
  "verdict",
  "rms",
  "views",
  "updated",
  "actions",
];

/** The columns a table `width` px wide keeps. The name, the verdict, the RMS
 * and the actions stay; the run and its views go first, then the updated
 * date, then the workflow. What a narrow table drops is in the name's
 * tooltip. `null` (not measured) keeps every column. */
export function visibleColumns(width: number | null): ReadonlySet<string> {
  if (width === null || width >= 900) return new Set(ALL_COLUMNS);
  const drop =
    width >= 700
      ? ["run", "views"]
      : width >= 560
        ? ["run", "views", "updated"]
        : ["run", "views", "updated", "workflow"];
  return new Set(ALL_COLUMNS.filter((k) => !drop.includes(k)));
}

/** What the columns of a narrow table drop say of `row`, as a line. */
export function rowDetails(row: ProjectRow): string {
  const parts = [
    row.entry.topology ? exportKindLabel(row.entry.topology) : "manifest unreadable",
  ];
  if (row.latest) {
    const of = row.entry.runs.length > 1 ? ` of ${row.entry.runs.length}` : "";
    parts.push(`latest run ${row.latest.id}${of}`);
  } else {
    parts.push("no run");
  }
  if (row.views) {
    parts.push(`${formatCount(row.views.usable)}/${formatCount(row.views.total)} views`);
  }
  if (row.updated !== null) parts.push(`updated ${dateOf(row.updated)}`);
  return parts.join(" · ");
}
