import { Callout, SkeletonRows } from "@vitavision/ui";
import type { DeviceSeriesSet, StudioError } from "../../api/generated/types";
import type { HistoryRow, XMode } from "./deviceModel";
import { HistoryTable } from "./HistoryTable";
import type { RunMeta } from "./seriesModel";
import { TrendsPanel } from "./TrendsPanel";

/** The Trends tab: the charts of every parameter, and (for one device) the
 * run list beside them, so a run hovered in either is marked in both. */
export function TrendsTab({
  series,
  error,
  rows,
  showRuns,
  meta,
  xMode,
  onXMode,
  activeKey,
  onHover,
  onSelect,
  onOpen,
  starredOnly,
}: {
  series: DeviceSeriesSet | undefined;
  error: StudioError | null;
  rows: HistoryRow[];
  /** The run list shows for one device; a type's units are lines. */
  showRuns: boolean;
  meta: ReadonlyMap<string, RunMeta>;
  xMode: XMode;
  onXMode: (mode: XMode) => void;
  activeKey: string | null;
  onHover: (key: string | null) => void;
  onSelect: (key: string) => void;
  onOpen: (row: HistoryRow) => void;
  starredOnly: boolean;
}) {
  return (
    <div
      className={
        showRuns
          ? "grid min-h-0 grid-cols-1 gap-4 xl:grid-cols-[minmax(0,26rem)_minmax(0,1fr)]"
          : "flex flex-col gap-3"
      }
    >
      {showRuns && (
        <div className="max-h-[70vh] min-w-0 overflow-y-auto rounded-panel border border-line bg-surface">
          <HistoryTable
            compact
            rows={rows}
            activeKey={activeKey}
            onHover={onHover}
            onSelect={onSelect}
            onOpen={onOpen}
          />
        </div>
      )}
      {error && series === undefined ? (
        <Callout tone="error" title="The parameters could not be read">
          {error.message}
        </Callout>
      ) : series === undefined ? (
        <div role="status" aria-busy aria-label="Reading the parameters">
          <SkeletonRows rows={5} />
        </div>
      ) : (
        <TrendsPanel
          series={series}
          meta={meta}
          xMode={xMode}
          onXMode={onXMode}
          hoverKey={activeKey}
          onHover={onHover}
          onPick={onSelect}
          starredOnly={starredOnly}
        />
      )}
    </div>
  );
}
