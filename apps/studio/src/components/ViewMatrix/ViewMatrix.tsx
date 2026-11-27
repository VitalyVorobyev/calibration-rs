import { StatusDot } from "@vitavision/ui";
import {
  SequenceMatrix,
  type SequenceMatrixColumn,
  type SequenceMatrixRow,
  type SequenceMatrixSelection,
} from "@vitavision/workbench";
import { ImageOff } from "lucide-react";
import { useMemo, type CSSProperties } from "react";
import type { ViewRow } from "../../api/generated/types";
import { hasGap, imagePath } from "./matrixModel";
import { Thumb } from "./Thumb";

export interface ViewMatrixProps {
  /** The views (rows), in order. */
  views: readonly ViewRow[];
  /** The cameras' ids (columns), in manifest order. */
  cameras: readonly string[];
  /** The folder the views' relative image paths resolve against. */
  dataRoot: string;
  "aria-label": string;
  value?: SequenceMatrixSelection | null | undefined;
  onValueChange?: ((value: SequenceMatrixSelection) => void) | undefined;
  onCellAction?: ((row: string, column: string) => void) | undefined;
  /** A cell's width and height, px (thumbnails keep 4:3). */
  colWidth?: number;
  rowHeight?: number;
  /** Size the matrix to its content, at most this many rows high (it then
   * scrolls inside); without it the matrix fills the box its parent gives. */
  fitRows?: number;
  className?: string | undefined;
}

/** The camera × view matrix of a dataset: a thumbnail per cell, a missing
 * image drawn as such, the view's token down the left. Only the cells in
 * view are drawn and their thumbnails queue behind a cap of six loads, so a
 * dataset of ten cameras and hundreds of views stays light. */
export function ViewMatrix({
  views,
  cameras,
  dataRoot,
  value,
  onValueChange,
  onCellAction,
  colWidth = 112,
  rowHeight = 72,
  fitRows,
  className,
  ...rest
}: ViewMatrixProps) {
  const rows = useMemo<SequenceMatrixRow[]>(
    () => views.map((v) => ({ id: v.token, label: v.token })),
    [views],
  );
  const columns = useMemo<SequenceMatrixColumn[]>(
    () => cameras.map((id) => ({ id, label: id })),
    [cameras],
  );
  const byToken = useMemo(() => new Map(views.map((v) => [v.token, v])), [views]);
  const cameraIndex = useMemo(() => new Map(cameras.map((c, i) => [c, i])), [cameras]);

  const fitStyle: CSSProperties | undefined =
    fitRows === undefined
      ? undefined
      : {
          // Header, the rows shown and the grid's border.
          height: 30 + Math.min(Math.max(views.length, 1), fitRows) * rowHeight + 2,
          width: 76 + cameras.length * colWidth + 2,
          maxWidth: "100%",
        };

  return (
    <SequenceMatrix
      aria-label={rest["aria-label"]}
      rows={rows}
      columns={columns}
      rowHeight={rowHeight}
      colWidth={colWidth}
      rowHeaderWidth={76}
      headerHeight={30}
      value={value}
      onValueChange={onValueChange}
      onCellAction={onCellAction}
      multiple={false}
      selectionFollowsFocus
      className={className}
      style={fitStyle}
      corner={<span className="text-[10px] tracking-wider uppercase">View</span>}
      renderRowHeader={(row) => {
        const view = byToken.get(row.id);
        return (
          <span className="flex min-w-0 items-center gap-1.5">
            {view && hasGap(view) && (
              <StatusDot tone="defect">
                <span className="sr-only">a camera has no image for this view</span>
              </StatusDot>
            )}
            <span className="truncate font-mono text-[11px] text-fg">{row.id}</span>
          </span>
        );
      }}
      renderColumnHeader={(column) => (
        <span className="truncate font-medium text-fg">{column.id}</span>
      )}
      cellLabel={(row, column) => {
        const view = byToken.get(row.id);
        const cam = cameraIndex.get(column.id);
        const image = view && cam !== undefined ? view.images[cam] : null;
        return `view ${row.id}, camera ${column.id}, ${image ?? "no image"}`;
      }}
      renderCell={(row, column) => {
        const view = byToken.get(row.id);
        const cam = cameraIndex.get(column.id);
        const path = view && cam !== undefined ? imagePath(dataRoot, view, cam) : null;
        if (path === null) {
          return (
            <span
              title="This camera has no image for this view"
              className="grid size-full place-items-center gap-0 bg-defect/8 text-defect"
            >
              <ImageOff aria-hidden className="size-4" />
            </span>
          );
        }
        return <Thumb path={path} />;
      }}
    />
  );
}
