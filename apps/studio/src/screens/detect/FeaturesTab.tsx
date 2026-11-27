import { Button, Callout, Empty, ReadoutStrip, Table } from "@vitavision/ui";
import { Ban, RotateCcw } from "lucide-react";
import { useMemo } from "react";
import type { DetectionDetail } from "../../api/generated/types";
import { formatCount } from "../../lib/format";
import { outcomeSentence } from "./detectModel";
import { featureColumns } from "./featureColumns";
import { featureRows } from "./overlayModel";

/** A PuzzleBoard whose read bits lose their vote this often is likely
 * labelled wrongly. */
const HIGH_DISSENT = 0.2;

/** Tables longer than this render only the rows in view. */
const WINDOW_ABOVE = 150;

/** What the detector found in the image, as a strip of facts and a table
 * of its features: hover and press are mirrored with the overlay, the
 * check column feeds the picked set that "Exclude" acts on. */
export function FeaturesTab({
  detail,
  expected,
  excluded,
  picked,
  hovered,
  onPick,
  onHover,
  onExclude,
  readOnly,
  unmatched,
}: {
  detail: DetectionDetail;
  expected: number | null;
  excluded: ReadonlySet<number>;
  picked: ReadonlySet<number>;
  hovered: number | null;
  onPick: (index: number, additive: boolean) => void;
  onHover: (index: number | null) => void;
  /** Exclude (or restore) the picked features. */
  onExclude: () => void;
  readOnly: boolean;
  /** How many excluded points of this image match no feature of it. */
  unmatched: number;
}) {
  const kind = detail.target.kind;
  const rows = useMemo(() => featureRows(detail), [detail]);
  const target = detail.target;
  const quality = target.kind === "ringgrid" ? null : target.quality;
  const pickedExcluded = [...picked].every((i) => excluded.has(i));

  const columns = useMemo(
    () => featureColumns(kind, picked, excluded, onPick),
    [kind, picked, excluded, onPick],
  );

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2.5">
      <ReadoutStrip
        items={[
          {
            label: "found",
            value: (
              <span className="font-mono">
                {formatCount(detail.features.length)}
                {expected ? ` of ${formatCount(expected)}` : ""}
              </span>
            ),
          },
          { label: "rejected", value: formatCount(detail.rejected?.length ?? 0) },
          target.kind === "chessboard" && target.cell_size_px != null
            ? { label: "cell", value: `${target.cell_size_px.toFixed(1)} px` }
            : { value: null },
          quality?.noise_sigma != null
            ? { label: "noise σ", value: quality.noise_sigma.toFixed(2) }
            : { value: null },
          quality?.local_residual_rms_px != null
            ? {
                label: "leave-one-out",
                value: `${quality.local_residual_rms_px.toFixed(3)} px`,
              }
            : { value: null },
        ]}
      />
      <p className="text-xs text-fg-muted">{outcomeSentence(detail.outcome)}</p>

      {target.kind === "puzzleboard" &&
        target.decode &&
        target.decode.dot_dissent_rate > HIGH_DISSENT && (
          <Callout tone="warning" title="The labelling is likely wrong">
            {(target.decode.dot_dissent_rate * 100).toFixed(0)}% of the read confidence
            lost its vote and {(target.decode.bit_error_rate * 100).toFixed(0)}% of the
            bits disagree with the master pattern.
          </Callout>
        )}
      {target.kind === "puzzleboard" &&
        target.decode &&
        target.decode.dot_dissent_rate <= HIGH_DISSENT && (
          <p className="text-[11px] text-fg-muted">
            Decode: bit error {(target.decode.bit_error_rate * 100).toFixed(1)}%, dot
            dissent {(target.decode.dot_dissent_rate * 100).toFixed(1)}%,{" "}
            {target.decode.edges_matched} of {target.decode.edges_observed} edges match
            the master.
          </p>
        )}
      {target.kind === "ringgrid" && !target.anchored && (
        <Callout tone="warning" title="Labels are relative">
          The view is not anchored on the board: the ring labels are relative to each
          other, not board positions.
        </Callout>
      )}
      {unmatched > 0 && (
        <Callout tone="warning">
          {unmatched} excluded {unmatched === 1 ? "point matches" : "points match"} no
          feature of this detection, so {unmatched === 1 ? "it leaves" : "they leave"}{" "}
          nothing out.
        </Callout>
      )}

      <div className="flex items-center gap-2">
        <Button
          size="sm"
          variant="secondary"
          disabled={readOnly || picked.size === 0}
          onClick={onExclude}
        >
          {pickedExcluded && picked.size > 0 ? (
            <RotateCcw aria-hidden className="size-3" />
          ) : (
            <Ban aria-hidden className="size-3" />
          )}
          {picked.size === 0
            ? "Exclude features"
            : pickedExcluded
              ? `Restore ${picked.size} ${picked.size === 1 ? "feature" : "features"}`
              : `Exclude ${picked.size} ${picked.size === 1 ? "feature" : "features"}`}
        </Button>
        <span className="text-[11px] text-fg-muted">
          Click a feature, ⇧-click to add, ⇧-drag to box.
        </span>
      </div>

      <div
        className={
          rows.length > WINDOW_ABOVE ? "min-h-0 flex-1" : "min-h-0 flex-1 overflow-y-auto"
        }
      >
        <Table
          caption="Features of this image"
          columns={columns}
          rows={rows}
          rowKey={(r) => r.index}
          {...(rows.length > WINDOW_ABOVE ? { windowed: { rowHeight: 24 } } : {})}
          isRowActive={(r) => hovered === r.index || picked.has(r.index)}
          onRowClick={(r, _i, e) => onPick(r.index, e.shiftKey || e.metaKey || e.ctrlKey)}
          onRowHover={(r) => onHover(r ? r.index : null)}
          empty={<Empty>No feature was kept in this image.</Empty>}
        />
      </div>
    </div>
  );
}
