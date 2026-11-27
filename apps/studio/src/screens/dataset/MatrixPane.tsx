import { Empty, Input, Select, SkeletonRows, ToggleChip } from "@vitavision/ui";
import type { SequenceMatrixSelection } from "@vitavision/workbench";
import { Search } from "lucide-react";
import { useMemo, useState } from "react";
import type { DatasetSpec, ProjectViews } from "../../api/generated/types";
import {
  filterViews,
  hasGap,
  NO_ROW_FILTER,
  type RowFilter,
} from "../../components/ViewMatrix/matrixModel";
import { ViewMatrix } from "../../components/ViewMatrix/ViewMatrix";
import type { Selection } from "../../lib/selection";

const ALL = "__all__";

/** The dataset as a camera × view matrix of thumbnails, with the filters
 * that find a bad view among hundreds: only the views with a missing image,
 * one camera, a token. The selection is the app's (a camera and a view), so
 * the keys that step views and cameras move it here too. */
export function MatrixPane({
  reply,
  loading,
  manifest,
  dataRoot,
  selection,
  onSelect,
}: {
  reply: ProjectViews | null;
  loading: boolean;
  manifest: DatasetSpec | null;
  dataRoot: string;
  selection: Selection;
  onSelect: (camera: number, view: number) => void;
}) {
  const [filter, setFilter] = useState<RowFilter>(NO_ROW_FILTER);
  const [camera, setCamera] = useState(ALL);

  const all = useMemo(() => reply?.views ?? [], [reply]);
  const views = useMemo(() => filterViews(all, filter), [all, filter]);
  const cameras = useMemo(
    () => (reply?.cameras ?? []).filter((c) => camera === ALL || c === camera),
    [reply, camera],
  );
  const problems = useMemo(() => all.filter(hasGap).length, [all]);

  // The matrix's selection is the app's: a camera and a view by index.
  const value = useMemo<SequenceMatrixSelection | null>(() => {
    const v = selection.view !== null ? all[selection.view] : undefined;
    const c = selection.camera !== null ? reply?.cameras[selection.camera] : undefined;
    return v && c ? { row: v.token, col: c } : null;
  }, [selection, all, reply]);

  const choose = (next: SequenceMatrixSelection) => {
    const view = all.findIndex((v) => v.token === next.row);
    const cam = reply?.cameras.indexOf(next.col) ?? -1;
    if (view >= 0 && cam >= 0) onSelect(cam, view);
  };

  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col gap-3">
      <div
        role="group"
        aria-label="Filter the views"
        className="flex flex-wrap items-center gap-2"
      >
        <ToggleChip
          size="sm"
          checked={filter.problemsOnly}
          disabled={problems === 0 && !filter.problemsOnly}
          onCheckedChange={(problemsOnly) => setFilter({ ...filter, problemsOnly })}
        >
          Problems only{problems > 0 ? ` · ${problems}` : ""}
        </ToggleChip>
        {(reply?.cameras.length ?? 0) > 1 && (
          <Select
            aria-label="Camera"
            className="w-36"
            value={camera}
            onValueChange={setCamera}
            options={[
              { value: ALL, label: "All cameras" },
              ...(reply?.cameras ?? []).map((c) => ({ value: c, label: c })),
            ]}
          />
        )}
        <div className="relative w-40">
          <Search
            aria-hidden
            className="pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2 text-fg-subtle"
          />
          <Input
            type="search"
            aria-label="Find a view by its token"
            placeholder="View token…"
            className="pl-8"
            value={filter.token}
            onChange={(e) => setFilter({ ...filter, token: e.target.value })}
          />
        </div>
        <span className="ml-auto font-mono text-xs text-fg-muted">
          {views.length === all.length
            ? `${all.length} ${all.length === 1 ? "view" : "views"}`
            : `${views.length} of ${all.length} views`}
        </span>
      </div>

      <div className="min-h-0 flex-1">
        {reply === null ? (
          <div
            role="status"
            aria-busy
            aria-label="Reading the layout"
            className="max-w-md"
          >
            <SkeletonRows rows={4} />
          </div>
        ) : all.length === 0 ? (
          <Empty className="py-10">
            {loading
              ? "Looking for the images…"
              : `Nothing matches ${patternsOf(manifest)}. Check the image patterns of the cameras.`}
          </Empty>
        ) : views.length === 0 ? (
          <Empty className="py-10">No view passes the filter.</Empty>
        ) : (
          <ViewMatrix
            aria-label="Camera by view"
            views={views}
            cameras={cameras}
            dataRoot={dataRoot}
            value={value}
            onValueChange={choose}
            className="h-full"
          />
        )}
      </div>
    </div>
  );
}

/** The cameras' patterns as one phrase: `cam0/*.png`, `cam1/*.png`. */
function patternsOf(manifest: DatasetSpec | null): string {
  const patterns = (manifest?.cameras ?? []).map((c) =>
    c.images.kind === "glob" ? `“${c.images.pattern}”` : "the list of images",
  );
  return patterns.length === 0 ? "the cameras" : [...new Set(patterns)].join(", ");
}
