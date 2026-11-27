import { imageViewBox, StageSurface, useStageHitTest } from "@vitavision/stage2d";
import { useState } from "react";
import { pointsInBox, type Box, type PickPoint } from "./pickModel";

/** The stage's one press target: a press on a feature picks it (⇧ or ⌘
 * adds to the picked), a ⇧-drag over bare image boxes the features under
 * the rectangle, and any other press pans. Only the accepted layers pick:
 * rejected candidates and marker quads are looked at, not excluded. */
export function PickSurface({
  points,
  image,
  onPick,
  onBox,
}: {
  points: readonly PickPoint[];
  image: { width: number; height: number };
  onPick: (index: number, additive: boolean) => void;
  onBox: (indices: number[], additive: boolean) => void;
}) {
  const { hitTest } = useStageHitTest();
  const [box, setBox] = useState<Box | null>(null);

  return (
    <>
      <StageSurface
        onPress={(press) => {
          const hit = hitTest(press.point, press.radius);
          const isFeature =
            hit !== null &&
            typeof hit.id === "number" &&
            hit.layerId.startsWith("acc") &&
            (hit.layerId.endsWith("corners") || hit.layerId.endsWith("rings"));
          if (isFeature) {
            return {
              onEnd: (_p, _e, moved) => {
                if (!moved) onPick(hit.id as number, press.shiftKey || press.metaKey);
              },
            };
          }
          if (!press.shiftKey) return undefined;
          const start = press.point;
          return {
            onMove: (p) => setBox({ x0: start.x, y0: start.y, x1: p.x, y1: p.y }),
            onEnd: (p, _e, moved) => {
              setBox(null);
              if (moved)
                onBox(
                  pointsInBox(points, { x0: start.x, y0: start.y, x1: p.x, y1: p.y }),
                  press.metaKey,
                );
            },
            onCancel: () => setBox(null),
          };
        }}
      />
      {box && (
        <svg
          aria-hidden
          viewBox={imageViewBox(image)}
          className="pointer-events-none absolute inset-0 size-full"
        >
          <rect
            x={Math.min(box.x0, box.x1)}
            y={Math.min(box.y0, box.y1)}
            width={Math.abs(box.x1 - box.x0)}
            height={Math.abs(box.y1 - box.y0)}
            className="fill-signal/10 stroke-signal"
            strokeWidth={1.5}
            strokeDasharray="4 3"
            vectorEffect="non-scaling-stroke"
          />
        </svg>
      )}
    </>
  );
}
