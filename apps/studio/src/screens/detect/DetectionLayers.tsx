import { EllipseSet, type EllipseSetItem } from "@vitavision/stage2d";
import {
  covarianceEllipse,
  TargetOverlay,
  type TargetHit,
  type TargetId,
} from "@vitavision/overlays";
import { useMemo } from "react";
import type { DetectionDetail } from "../../api/generated/types";
import { acceptedDetection, rejectedDetection, type RejectedItem } from "./overlayModel";
import type { LayerState } from "./layers";

/** The gain of the position-uncertainty ellipses: a detector's σ is a
 * fraction of a pixel, so they are drawn this many times larger and the
 * stage says so. */
export const ELLIPSE_GAIN = 20;

/** What of a detection the stage draws: the accepted features with the
 * target's own marks (lattice, marker quads, edge bits, ring ellipses), the
 * rejected candidates dimmed under them, the excluded features crossed, and
 * the uncertainty ellipses. Picking is the surface's (`PickSurface`); this
 * reports what the pointer is over. */
export function DetectionLayers({
  detail,
  rejected,
  layers,
  hovered,
  picked,
  excluded,
  onHover,
}: {
  detail: DetectionDetail;
  rejected: readonly RejectedItem[];
  layers: LayerState;
  hovered: TargetId | null;
  picked: ReadonlySet<number>;
  /** Indices of the features the project excludes. */
  excluded: ReadonlySet<number>;
  onHover: (hit: TargetHit | null) => void;
}) {
  const accepted = useMemo(
    () => acceptedDetection(detail, { axes: layers.axes, excluded }),
    [detail, layers.axes, excluded],
  );
  const rejectedDet = useMemo(() => rejectedDetection(rejected), [rejected]);
  const ringCov = useMemo<EllipseSetItem[]>(() => {
    if (!layers.uncertainty || detail.target.kind !== "ringgrid") return [];
    return detail.target.rings.flatMap((ring, index) => {
      const f = detail.features[index];
      const e = ring.center_cov_px
        ? covarianceEllipse(ring.center_cov_px, ELLIPSE_GAIN)
        : null;
      return f && e
        ? [
            {
              id: index,
              x: f.image_xy[0],
              y: f.image_xy[1],
              rx: e.rx,
              ry: e.ry,
              angle: e.angle,
              role: "model" as const,
            },
          ]
        : [];
    });
  }, [layers.uncertainty, detail]);

  const uncertainty =
    layers.uncertainty && detail.target.kind !== "ringgrid"
      ? { scale: ELLIPSE_GAIN }
      : undefined;

  return (
    <>
      {layers.rejected && rejected.length > 0 && (
        <TargetOverlay
          detection={rejectedDet}
          dimmed
          showEdges={false}
          showLabels={layers.labels}
          layerIdPrefix="rej"
        />
      )}
      {layers.accepted && (
        <TargetOverlay
          detection={accepted}
          showEdges={layers.edges}
          showLabels={layers.labels}
          hoveredId={hovered}
          selectedIds={picked}
          onHoverChange={onHover}
          layerIdPrefix="acc"
          {...(uncertainty ? { uncertainty } : {})}
        />
      )}
      {ringCov.length > 0 && <EllipseSet items={ringCov} pickable={false} role="model" />}
    </>
  );
}
