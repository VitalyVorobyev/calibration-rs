import {
  PointMarquee,
  StageSurface,
  useStageHitTest,
  type PointItem,
  type PointMarqueeHandle,
} from "@vitavision/stage2d";
import { useRef } from "react";

/** The stage's one press target: a press on a feature picks it (⇧ or ⌘
 * adds to the picked), a ⇧-drag over bare image boxes the features under
 * the rectangle (`PointMarquee`; ⌘ as well adds, Escape cancels), and any
 * other press pans. Only the accepted layers pick: rejected candidates and
 * marker quads are looked at, not excluded. */
export function PickSurface({
  points,
  onPick,
  onBox,
}: {
  points: readonly PointItem[];
  onPick: (index: number, additive: boolean) => void;
  onBox: (indices: number[], additive: boolean) => void;
}) {
  const { hitTest } = useStageHitTest();
  const marquee = useRef<PointMarqueeHandle>(null);

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
          return marquee.current?.sweepDrag(press);
        }}
      />
      <PointMarquee
        ref={marquee}
        items={points}
        surface={false}
        onSelect={(ids, mode) =>
          onBox(
            ids.filter((id): id is number => typeof id === "number"),
            mode === "add",
          )
        }
      />
    </>
  );
}
