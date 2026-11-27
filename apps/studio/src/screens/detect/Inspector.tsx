import { Callout, Empty, ErrorBox, SkeletonRows, Tabs } from "@vitavision/ui";
import { useState, type ComponentProps } from "react";
import type { Query } from "../../session/query";
import type { DetectionDetail, ImageDetection } from "../../api/generated/types";
import { DetectorTab } from "./DetectorTab";
import { FeaturesTab } from "./FeaturesTab";
import { RejectedTab } from "./RejectedTab";
import type { RejectedItem } from "./overlayModel";

type TabId = "features" | "rejected" | "detector";

/** The right side: what the detector found in the image (Features), what it
 * left out and why (Rejected), and the detector's settings to try (Detector).
 * The tab stays when the image changes. */
export function Inspector({
  detection,
  shown,
  detailQuery,
  expected,
  rejected,
  features,
  onFocus,
  detector,
}: {
  detection: ImageDetection | null;
  /** The detection the stage shows (cached or preview). */
  shown: DetectionDetail | null;
  detailQuery: Query<DetectionDetail>;
  expected: number | null;
  rejected: readonly RejectedItem[];
  features: Omit<ComponentProps<typeof FeaturesTab>, "detail" | "expected">;
  onFocus: (item: RejectedItem) => void;
  detector: ComponentProps<typeof DetectorTab>;
}) {
  const [tab, setTab] = useState<TabId>("features");
  const items: { id: TabId; label: string }[] = [
    { id: "features", label: "Features" },
    {
      id: "rejected",
      label: `Rejected${shown && rejected.length > 0 ? ` · ${rejected.length}` : ""}`,
    },
    { id: "detector", label: "Detector" },
  ];

  const nothing = (() => {
    if (detection === null)
      return <Empty className="py-10">Select an image to read its detection.</Empty>;
    if (shown) return null;
    if (detection.state.kind === "failed")
      return (
        <ErrorBox>
          Detection of this image failed: {detection.state.error.message}
        </ErrorBox>
      );
    if (detailQuery.error) return <ErrorBox>{detailQuery.error.message}</ErrorBox>;
    if (detailQuery.loading)
      return (
        <div role="status" aria-busy aria-label="Reading the detection">
          <SkeletonRows rows={5} />
        </div>
      );
    return (
      <Callout tone="info" title="No detection to show">
        This image has not been detected with the project's detector settings. Detect it
        from the stage, or detect all images from the matrix.
      </Callout>
    );
  })();

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3">
      <Tabs
        items={items}
        active={tab}
        onSelect={setTab}
        label="Inspector"
        idPrefix="detect-inspector"
      />
      <div
        role="tabpanel"
        id={`detect-inspector-panel-${tab}`}
        aria-labelledby={`detect-inspector-tab-${tab}`}
        className="flex min-h-0 flex-1 flex-col"
      >
        {tab === "detector" ? (
          <DetectorTab {...detector} />
        ) : nothing ? (
          nothing
        ) : shown && tab === "features" ? (
          <FeaturesTab detail={shown} expected={expected} {...features} />
        ) : shown ? (
          <RejectedTab
            detail={shown}
            items={rejected}
            leftOut={detection?.left_out}
            onFocus={onFocus}
          />
        ) : null}
      </div>
    </div>
  );
}
