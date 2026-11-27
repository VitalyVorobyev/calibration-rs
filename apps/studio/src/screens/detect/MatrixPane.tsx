import {
  Button,
  Callout,
  Empty,
  ErrorBox,
  Input,
  Progress,
  Select,
  SkeletonRows,
  ToggleChip,
} from "@vitavision/ui";
import type { SequenceMatrixSelection } from "@vitavision/workbench";
import { Ban, Play, Search, Square } from "lucide-react";
import { useMemo, useState } from "react";
import type { DetectionOverview, Exclusions, ViewRow } from "../../api/generated/types";
import { ViewMatrix } from "../../components/ViewMatrix/ViewMatrix";
import { formatCount } from "../../lib/format";
import type { Selection } from "../../lib/selection";
import { describeJob } from "../../session/jobModel";
import { useReadOnly } from "../../session/jobs";
import { CountCell } from "./CountCell";
import {
  cellStatus,
  filterRows,
  isFiltering,
  neverDetected,
  NO_DETECT_FILTER,
  overviewStats,
  type DetectFilter,
} from "./detectModel";
import { isImageOut, isViewExcluded } from "./exclusions";
import type { useDetectAll } from "./useDetectData";

/** The mark after the token of a view the project excludes. */
function excludedMark(out: boolean) {
  return out ? (
    <span title="Excluded from the calibration">
      <Ban aria-hidden className="size-3 text-fg-subtle" />
      <span className="sr-only">view excluded</span>
    </span>
  ) : null;
}

const COUNT_BELOW = ["any", "10", "20", "30", "50"];

/** The project's images as the detection cache knows them: a camera ×
 * view matrix of feature counts with the filters that find a bad image
 * among hundreds, and the job that fills the cache. */
export function MatrixPane({
  overview,
  loading,
  error,
  exclude,
  dataRoot,
  selection,
  onSelect,
  detectAll,
}: {
  overview: DetectionOverview | undefined;
  loading: boolean;
  error: string | null;
  exclude: Exclusions;
  dataRoot: string;
  selection: Selection;
  onSelect: (camera: number, view: number) => void;
  detectAll: ReturnType<typeof useDetectAll>;
}) {
  const readOnly = useReadOnly();
  const [filter, setFilter] = useState<DetectFilter>(NO_DETECT_FILTER);
  const stats = useMemo(() => (overview ? overviewStats(overview) : null), [overview]);

  const expected = overview?.expected ?? null;
  const statusOf = useMemo(
    () => (view: DetectionOverview["views"][number], camera: number) => {
      const image = view.images[camera] ?? null;
      const excluded =
        image !== null &&
        isImageOut(
          exclude,
          { token: view.token, images: view.images.map((i) => i?.image ?? null) },
          overview?.cameras[camera] ?? "",
          image.image,
        );
      return cellStatus(image, expected, excluded);
    },
    [exclude, expected, overview],
  );

  const rows = useMemo(
    () => (overview ? filterRows(overview.views, filter, statusOf) : []),
    [overview, filter, statusOf],
  );
  const viewRows = useMemo<ViewRow[]>(
    () =>
      rows.map(({ view }) => ({
        token: view.token,
        images: view.images.map((i) => i?.image ?? null),
      })),
    [rows],
  );
  const byToken = useMemo(
    () => new Map((overview?.views ?? []).map((v) => [v.token, v])),
    [overview],
  );

  const value = useMemo<SequenceMatrixSelection | null>(() => {
    const view = selection.view !== null ? overview?.views[selection.view] : undefined;
    const camera =
      selection.camera !== null ? overview?.cameras[selection.camera] : undefined;
    return view && camera ? { row: view.token, col: camera } : null;
  }, [selection, overview]);

  const choose = (next: SequenceMatrixSelection) => {
    const view = overview?.views.findIndex((v) => v.token === next.row) ?? -1;
    const camera = overview?.cameras.indexOf(next.col) ?? -1;
    if (view >= 0 && camera >= 0) onSelect(camera, view);
  };

  const line = detectAll.row ? describeJob(detectAll.row) : null;
  const never = stats !== null && neverDetected(stats);

  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col gap-2.5">
      <div className="flex flex-col gap-2">
        {detectAll.running ? (
          <div className="flex flex-col gap-1.5 rounded-control border border-line bg-surface p-2.5">
            <div className="flex items-center gap-2">
              <span role="status" className="text-xs font-medium text-fg">
                {line?.stage ?? "Starting detection"}
              </span>
              <Button
                size="sm"
                variant="secondary"
                className="ml-auto"
                disabled={!detectAll.row || detectAll.row.cancelling}
                onClick={detectAll.cancel}
              >
                <Square aria-hidden className="size-3" />
                {detectAll.row?.cancelling ? "Cancelling…" : "Cancel"}
              </Button>
            </div>
            <Progress
              aria-label="Detection progress"
              value={line?.progress ? line.progress.done : null}
              max={line?.progress ? line.progress.total : 100}
            />
            <p className="text-[11px] text-fg-muted">
              Counts fill in as it goes. A cancelled run keeps what it detected.
            </p>
          </div>
        ) : (
          <div className="flex items-center gap-2">
            {stats && stats.pending === 0 && stats.images > 0 ? (
              <Button
                size="sm"
                variant="secondary"
                disabled={readOnly}
                title="Detect every image again, ignoring the cache"
                onClick={() => void detectAll.start(true)}
              >
                <Play aria-hidden className="size-3" />
                Detect again
              </Button>
            ) : (
              <Button
                size="sm"
                variant={never ? "primary" : "secondary"}
                disabled={readOnly || overview === undefined}
                onClick={() => void detectAll.start(false)}
              >
                <Play aria-hidden className="size-3" />
                Detect all
              </Button>
            )}
            {stats && stats.analysed > 0 && stats.pending > 0 && (
              <span className="text-[11px] text-fg-muted">
                {formatCount(stats.analysed)} of {formatCount(stats.images)} images
                analysed
              </span>
            )}
          </div>
        )}
        {detectAll.error && (
          <ErrorBox>
            Detection stopped: {detectAll.error.message}{" "}
            <button
              type="button"
              className="underline"
              onClick={() => detectAll.clearError()}
            >
              Dismiss
            </button>
          </ErrorBox>
        )}
        {never && !detectAll.running && (
          <Callout tone="info">
            Nothing has been detected with these detector settings yet. Run detection to
            fill in the counts; the matrix shows <span className="font-mono">·</span>{" "}
            until then.
          </Callout>
        )}
      </div>

      <div
        role="group"
        aria-label="Filter the views"
        className="flex flex-wrap items-center gap-1.5"
      >
        <ToggleChip
          size="sm"
          checked={filter.problemsOnly}
          onCheckedChange={(problemsOnly) => setFilter({ ...filter, problemsOnly })}
        >
          Only problems
        </ToggleChip>
        <ToggleChip
          size="sm"
          checked={filter.excludedOnly}
          onCheckedChange={(excludedOnly) => setFilter({ ...filter, excludedOnly })}
        >
          Only excluded
        </ToggleChip>
        <Select
          aria-label="Fewer features than"
          className="w-32"
          value={filter.countBelow === null ? "any" : String(filter.countBelow)}
          onValueChange={(v) =>
            setFilter({ ...filter, countBelow: v === "any" ? null : Number(v) })
          }
          options={COUNT_BELOW.map((v) => ({
            value: v,
            label: v === "any" ? "Any count" : `count < ${v}`,
          }))}
        />
        <div className="relative w-28">
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
        <span className="ml-auto font-mono text-[11px] text-fg-muted">
          {overview === undefined
            ? ""
            : rows.length === overview.views.length
              ? `${overview.views.length} ${overview.views.length === 1 ? "view" : "views"}`
              : `${rows.length} of ${overview.views.length} views`}
        </span>
      </div>

      <div className="min-h-0 flex-1">
        {error !== null && overview === undefined ? (
          <ErrorBox>The detections cannot be listed: {error}</ErrorBox>
        ) : overview === undefined ? (
          <div role="status" aria-busy aria-label="Reading the detection cache">
            <SkeletonRows rows={5} />
          </div>
        ) : overview.views.length === 0 ? (
          <Empty className="py-10">
            The dataset has no views, so there is nothing to detect. Check its image
            patterns on the Dataset screen.
          </Empty>
        ) : viewRows.length === 0 ? (
          <Empty className="py-10">
            {isFiltering(filter) ? "No view passes the filter." : "No views."}
          </Empty>
        ) : (
          <div
            className={loading ? "size-full opacity-70 transition-opacity" : "size-full"}
          >
            <ViewMatrix
              aria-label="Detections by camera and view"
              views={viewRows}
              cameras={overview.cameras}
              dataRoot={dataRoot}
              value={value}
              onValueChange={choose}
              rowHeight={40}
              colWidth={72}
              className="h-full"
              describeCell={(view, camera) => {
                const detections = byToken.get(view.token);
                const status = detections ? statusOf(detections, camera) : null;
                return `view ${view.token}, camera ${overview.cameras[camera]}, ${status?.text ?? ""}: ${status?.sentence ?? ""}`;
              }}
              renderCellContent={(view, camera) => {
                const detections = byToken.get(view.token);
                return detections ? (
                  <CountCell status={statusOf(detections, camera)} />
                ) : null;
              }}
              rowMark={(view) =>
                excludedMark(
                  byToken.has(view.token) &&
                    isViewExcluded(exclude, { token: view.token, images: view.images }),
                )
              }
            />
          </div>
        )}
      </div>
    </div>
  );
}
