import {
  ImageLayer,
  ImageStage,
  StageLayersMenu,
  StageReadout,
  StageToolbar,
  StageToolbarDivider,
  type StageHandle,
  type StageView,
} from "@vitavision/stage2d";
import type { TargetHit, TargetId } from "@vitavision/overlays";
import { Button, Callout, Empty, ErrorBox } from "@vitavision/ui";
import { Ban, ScanSearch } from "lucide-react";
import { useMemo, useState, type ReactNode, type Ref } from "react";
import { api } from "../../api";
import type { DetectionDetail, DetectionOverview } from "../../api/generated/types";
import { useImageSize } from "../../hooks/useImageSize";
import type { Query } from "../../session/query";
import { DetectionLayers, ELLIPSE_GAIN } from "./DetectionLayers";
import { DEFAULT_LAYERS, LAYER_MENU, type LayerState } from "./layers";
import { PickSurface } from "./PickSurface";
import type { Cell } from "./cell";
import type { RejectedItem } from "./overlayModel";

/** A callout over the stage: the stage is dark in both themes, so the
 * callout's translucent tone sits on a surface of its own. */
function StageNote({ children }: { children: ReactNode }) {
  return <div className="rounded-panel bg-surface shadow-sm">{children}</div>;
}

/** The selected image in a zoomable stage with the detection over it, and
 * what to say when there is nothing to draw: no image, an image that does
 * not open, a detection not made yet, one being read, one that failed. */
export function StagePane({
  cell,
  detail,
  shown,
  rejected,
  detecting,
  needsDetect,
  onDetect,
  excluded,
  imageExcluded,
  picked,
  hovered,
  onHover,
  onPick,
  onBox,
  onClear,
  stageRef,
  readOnly,
}: {
  cell: Cell | null;
  detail: Query<DetectionDetail>;
  /** The detection drawn: the cached one, or a preview. */
  shown: DetectionDetail | null;
  rejected: readonly RejectedItem[];
  /** The image is being detected (a detail was asked for). */
  detecting: boolean;
  /** The image has no cached detection and none was asked for. */
  needsDetect: boolean;
  onDetect: () => void;
  excluded: ReadonlySet<number>;
  /** The image (or its view) is left out of the calibration. */
  imageExcluded: boolean;
  picked: ReadonlySet<number>;
  hovered: number | null;
  onHover: (hit: TargetHit | null) => void;
  onPick: (index: number, additive: boolean) => void;
  onBox: (indices: number[], additive: boolean) => void;
  onClear: () => void;
  stageRef: Ref<StageHandle>;
  readOnly: boolean;
}) {
  const [layers, setLayers] = useState<LayerState>(DEFAULT_LAYERS);
  // The zoom and pan stay while stepping through images of one size, and
  // open fitted on an image of another.
  const [viewOf, setViewOf] = useState<{ size: string; view: StageView | null }>({
    size: "",
    view: null,
  });
  const [cursor, setCursor] = useState<{ x: number; y: number } | null>(null);
  const url = cell?.path ? api.urls.image({ path: cell.path }) : null;
  const size = useImageSize(url);

  const points = useMemo(
    () =>
      (shown?.features ?? []).map((f, index) => ({
        id: index,
        x: f.image_xy[0],
        y: f.image_xy[1],
      })),
    [shown],
  );

  if (cell === null) {
    return (
      <Empty className="py-16">
        Select an image in the matrix to see what the detector found in it.
      </Empty>
    );
  }
  if (cell.path === null || cell.image === null) {
    return (
      <Empty className="py-16">
        {cell.cameraId} has no image for view {cell.token}.
      </Empty>
    );
  }
  if (size === "error") {
    return (
      <ErrorBox className="m-3">
        The image does not open:{" "}
        <span className="font-mono text-[11px] break-all">{cell.path}</span>
      </ErrorBox>
    );
  }
  if (size === null) {
    return (
      <div
        role="status"
        aria-busy
        aria-label="Loading the image"
        className="size-full animate-pulse rounded-control bg-raised"
      />
    );
  }

  const sizeKey = `${size.width}x${size.height}`;
  const banner = (
    <div className="flex max-w-[22rem] flex-col gap-2">
      {imageExcluded && (
        <StageNote>
          <Callout tone="warning" title="Excluded from calibration">
            <span className="inline-flex items-center gap-1.5">
              <Ban aria-hidden className="size-3.5" />
              This image is left out of every run. Its features stay drawn so they can be
              found.
            </span>
          </Callout>
        </StageNote>
      )}
      {needsDetect && (
        <StageNote>
          <Callout
            tone="info"
            title="Not detected yet"
            actions={
              <Button size="sm" disabled={readOnly} onClick={onDetect}>
                <ScanSearch aria-hidden className="size-3.5" />
                Detect this image
              </Button>
            }
          >
            This image has no detection with the project's detector settings. Detecting it
            stores the result with the others.
          </Callout>
        </StageNote>
      )}
      {detecting && detail.loading && shown === null && (
        <StageNote>
          <Callout tone="info" title="Detecting…">
            Reading the detection of this image.
          </Callout>
        </StageNote>
      )}
      {detail.error && shown === null && (
        <StageNote>
          <Callout tone="error" title="The detection failed">
            {detail.error.message}
          </Callout>
        </StageNote>
      )}
    </div>
  );

  const hoveredId: TargetId | null = hovered;
  return (
    <div className="relative size-full min-h-0 select-none">
      <ImageStage
        key={url}
        image={size}
        view={viewOf.size === sizeKey ? viewOf.view : null}
        onView={(next) => setViewOf({ size: sizeKey, view: next })}
        initialView="fit"
        label={`Image ${cell.image}`}
        className="bg-canvas"
        ref={stageRef}
        onHover={setCursor}
        onBackgroundClick={onClear}
        banner={banner}
        toolbar={
          <StageToolbar>
            <StageToolbarDivider />
            <StageLayersMenu
              layers={LAYER_MENU.map((l) => ({
                id: l.id,
                label: l.label,
                visible: layers[l.id],
              }))}
              onVisibleChange={(id, visible) => setLayers({ ...layers, [id]: visible })}
            />
          </StageToolbar>
        }
        readout={
          <StageReadout
            cursor={cursor}
            extra={layers.uncertainty ? `ellipses ×${ELLIPSE_GAIN}` : undefined}
          />
        }
      >
        {layers.image && <ImageLayer src={url ?? ""} alt="" pixelatedAbove={1} />}
        {shown && (
          <>
            <DetectionLayers
              detail={shown}
              rejected={rejected}
              layers={layers}
              hovered={hoveredId}
              picked={picked}
              excluded={excluded}
              onHover={onHover}
            />
            <PickSurface points={points} onPick={onPick} onBox={onBox} />
          </>
        )}
      </ImageStage>
    </div>
  );
}

export type { DetectionOverview };
