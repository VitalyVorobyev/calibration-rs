import { Badge, Callout, Empty } from "@vitavision/ui";
import { useMemo } from "react";
import type { DetectionDetail, RejectReason } from "../../api/generated/types";
import { leftOutSentence, outcomeSentence, rejectionTitle } from "./detectModel";
import type { RejectedItem } from "./overlayModel";

/** Why the detector left candidates out, grouped by reason and said in
 * words, with the image's own verdict above (what the detector made of the
 * whole image, and what the last run did with it). A candidate is pressed
 * to be shown on the image. */
export function RejectedTab({
  detail,
  items,
  leftOut,
  onFocus,
}: {
  detail: DetectionDetail;
  items: readonly RejectedItem[];
  /** Why the overview's run left this image out. */
  leftOut: RejectReason | null | undefined;
  onFocus: (item: RejectedItem) => void;
}) {
  const groups = useMemo(() => {
    const out = new Map<RejectedItem["group"], RejectedItem[]>();
    for (const item of items) out.set(item.group, [...(out.get(item.group) ?? []), item]);
    return [...out.entries()];
  }, [items]);
  const outcome = detail.outcome;
  const notice = outcome.status !== "detected";

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto">
      {notice && (
        <Callout tone="warning" title="The detector's verdict on this image">
          {outcomeSentence(outcome)}
        </Callout>
      )}
      {leftOut && (
        <Callout tone="warning" title="The last run left this image out">
          {leftOutSentence(leftOut)}
        </Callout>
      )}
      {items.length === 0 ? (
        <Empty>
          Nothing was rejected in this image: every candidate the detector saw was kept.
        </Empty>
      ) : (
        groups.map(([group, list]) => (
          <section
            key={group}
            aria-label={groupTitle(group)}
            className="flex flex-col gap-1"
          >
            <h3 className="flex items-center gap-2 text-[10px] font-semibold tracking-wider text-fg-muted uppercase">
              {groupTitle(group)}
              <Badge tone="neutral">{list.length}</Badge>
            </h3>
            <p className="text-xs text-fg-muted">
              {group === "marker"
                ? "Marker cells whose bits match another marker better than the one the board places there."
                : `${list[0]!.sentence}.`}
            </p>
            <ul className="flex flex-col">
              {list.map((item) => (
                <li key={item.id}>
                  <button
                    type="button"
                    onClick={() => onFocus(item)}
                    className="flex w-full items-center gap-2 rounded-control px-2 py-1 text-left font-mono text-[11px] text-fg outline-none hover:bg-raised focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-signal"
                  >
                    <span className="min-w-0 flex-1 truncate">
                      {item.x.toFixed(1)}, {item.y.toFixed(1)}
                      {item.label ? ` · ${item.label}` : ""}
                    </span>
                    {item.marker && (
                      <span className="text-fg-muted">{item.sentence}</span>
                    )}
                    {item.point?.reason.kind === "unlabelled" && (
                      <span className="text-fg-muted">
                        response {item.point.reason.strength.toFixed(2)}
                      </span>
                    )}
                    {item.point &&
                      (item.point.reason.kind === "unanchored" ||
                        item.point.reason.kind === "uncorrected_center") && (
                        <span className="text-fg-muted">
                          conf. {item.point.reason.confidence.toFixed(2)}
                        </span>
                      )}
                  </button>
                </li>
              ))}
            </ul>
          </section>
        ))
      )}
    </div>
  );
}

function groupTitle(group: RejectedItem["group"]): string {
  return group === "marker" ? "Marker misread" : rejectionTitle(group);
}
