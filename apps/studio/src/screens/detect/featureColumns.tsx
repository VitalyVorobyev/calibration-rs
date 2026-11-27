import { Badge, Checkbox, type Column } from "@vitavision/ui";
import { featureLabel, type FeatureRow, type TargetKindName } from "./overlayModel";

const CENTER_WORD = {
  outer_ellipse: "outer ellipse",
  projective: "projective",
  photometric: "photometric",
} as const;

/** The Features table's columns for a target kind: a check column that feeds
 * the picked set, the feature's label, its position, its score or (rings)
 * confidence and centre estimator, its σ, and its status. */
export function featureColumns(
  kind: TargetKindName,
  picked: ReadonlySet<number>,
  excluded: ReadonlySet<number>,
  onPick: (index: number, additive: boolean) => void,
): Column<FeatureRow>[] {
  const ring = kind === "ringgrid";
  const cols: Column<FeatureRow>[] = [
    {
      key: "pick",
      header: <span className="sr-only">Picked</span>,
      width: "2rem",
      cell: (r) => (
        <Checkbox
          aria-label={`Pick feature ${featureLabel(kind, r)}`}
          checked={picked.has(r.index)}
          onCheckedChange={() => onPick(r.index, true)}
          onClick={(e) => e.stopPropagation()}
        />
      ),
    },
    {
      key: "label",
      header: kind === "puzzleboard" ? "Id" : ring ? "Code" : "i,j",
      width: "3.6rem",
      sortValue: (r) => r.index,
      cell: (r) => <span className="font-mono">{featureLabel(kind, r)}</span>,
    },
    {
      key: "x",
      header: "x",
      numeric: true,
      width: "4.4rem",
      sortValue: (r) => r.x,
      cell: (r) => r.x.toFixed(2),
    },
    {
      key: "y",
      header: "y",
      numeric: true,
      width: "4.4rem",
      sortValue: (r) => r.y,
      cell: (r) => r.y.toFixed(2),
    },
    ring
      ? {
          key: "confidence",
          header: "Conf.",
          numeric: true,
          width: "3.6rem",
          firstSort: "descending",
          sortValue: (r) => r.confidence,
          cell: (r) => (r.confidence === null ? "–" : r.confidence.toFixed(2)),
        }
      : {
          key: "score",
          header: "Score",
          numeric: true,
          width: "3.6rem",
          firstSort: "descending",
          sortValue: (r) => r.score,
          cell: (r) => (r.score === null ? "–" : r.score.toFixed(2)),
        },
    ...(ring
      ? [
          {
            key: "center",
            header: "Centre",
            width: "5.6rem",
            sortValue: (r: FeatureRow) => r.center,
            cell: (r: FeatureRow) => (r.center ? CENTER_WORD[r.center] : "–"),
          },
        ]
      : []),
    {
      key: "sigma",
      header: <span className="whitespace-nowrap">σ px</span>,
      numeric: true,
      width: "4.2rem",
      sortValue: (r) => r.sigmaPx,
      cell: (r) => (r.sigmaPx === null ? "–" : r.sigmaPx.toFixed(3)),
    },
    {
      key: "status",
      header: "Status",
      width: "6rem",
      cell: (r) =>
        excluded.has(r.index) ? (
          <Badge tone="neutral">excluded</Badge>
        ) : r.masterMismatch ? (
          <Badge tone="warning">master mismatch</Badge>
        ) : (
          <span className="text-fg-muted">accepted</span>
        ),
    },
  ];
  return cols;
}
