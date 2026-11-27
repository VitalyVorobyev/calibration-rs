import { ReadoutStrip } from "@vitavision/ui";
import { featureLabel, type FeatureRow, type TargetKindName } from "./overlayModel";

const CENTER = {
  outer_ellipse: "outer ellipse",
  projective: "projective",
  photometric: "photometric",
} as const;

/** One feature in a line under the stage: the one the pointer is over, else
 * the last one picked; a hint to hover when there is none. */
export function FeatureReadout({
  kind,
  row,
  excluded,
}: {
  kind: TargetKindName | null;
  row: FeatureRow | null;
  excluded: boolean;
}) {
  if (kind === null || row === null) {
    return (
      <p className="h-5 truncate text-[11px] leading-5 text-fg-subtle">
        Hover a feature to read it; press it to pick it.
      </p>
    );
  }
  return (
    <ReadoutStrip
      className="h-5 overflow-hidden"
      items={[
        {
          label: kind === "puzzleboard" ? "id" : "feature",
          value: featureLabel(kind, row),
        },
        { label: "x", value: row.x.toFixed(2) },
        { label: "y", value: row.y.toFixed(2) },
        row.score !== null
          ? { label: "score", value: row.score.toFixed(2) }
          : { value: null },
        row.confidence !== null
          ? { label: "confidence", value: row.confidence.toFixed(2) }
          : { value: null },
        row.center ? { label: "centre", value: CENTER[row.center] } : { value: null },
        row.sigmaPx !== null
          ? { label: "σ", value: `${row.sigmaPx.toFixed(3)} px` }
          : { value: null },
        row.masterMismatch ? { value: "master mismatch" } : { value: null },
        excluded ? { value: "excluded" } : { value: null },
      ]}
    />
  );
}
